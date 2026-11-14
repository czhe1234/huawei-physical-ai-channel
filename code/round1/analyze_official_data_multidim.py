#!/usr/bin/env python3
"""Multidimensional analysis of the official Round1 data."""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.stats import ks_2samp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from generate_v87_easy_field_projection import summarize
from powerfield_common import BS_POS, regime_of
from project_paths import PATHS


OUT_DIR = PATHS.analysis / "results" / "official_data_multidim"
REPORT = OUT_DIR / "official_data_multidim_analysis.json"
GEOMETRY_SHIFT_CSV = OUT_DIR / "official_geometry_train_test_shift.csv"
SPATIAL_CORR_CSV = OUT_DIR / "official_channel_spatial_correlations.csv"
REGIME_CHANNEL_CSV = OUT_DIR / "official_channel_by_regime.csv"

TINY = 1e-12
REGIMES = ("clear", "light", "medium", "heavy")


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0}
    return summarize(values)


def entropy01(probability: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    p = np.maximum(probability, TINY)
    raw = -np.sum(p * np.log(p), axis=axis)
    if isinstance(axis, tuple):
        size = np.prod([probability.shape[item] for item in axis])
    else:
        size = probability.shape[axis]
    return (raw / np.log(size)).astype(np.float32)


def psi(train: np.ndarray, test: np.ndarray, bins: int = 10) -> float:
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    edges = np.quantile(train, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    train_hist, _ = np.histogram(train, edges)
    test_hist, _ = np.histogram(test, edges)
    train_rate = np.maximum(train_hist / max(train_hist.sum(), 1), 1e-6)
    test_rate = np.maximum(test_hist / max(test_hist.sum(), 1), 1e-6)
    return float(np.sum((test_rate - train_rate) * np.log(test_rate / train_rate)))


def read_ply_header(path: Path) -> tuple[int, int, list[str]]:
    with path.open("rb") as handle:
        offset = 0
        vertex_count = 0
        lines = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY header ended unexpectedly")
            offset += len(line)
            text = line.decode("ascii").strip()
            lines.append(text)
            if text.startswith("element vertex"):
                vertex_count = int(text.split()[-1])
            if text == "end_header":
                break
    return offset, vertex_count, lines


def pointcloud_stats() -> dict[str, object]:
    path = PATHS.round1 / "Round1_Map.ply"
    offset, vertex_count, header = read_ply_header(path)
    dtype = np.dtype(
        [
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("nx", "<f8"),
            ("ny", "<f8"),
            ("nz", "<f8"),
        ]
    )
    points = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(vertex_count,))
    xyz = np.column_stack([points["x"], points["y"], points["z"]]).astype(np.float64)
    normal = np.column_stack([points["nx"], points["ny"], points["nz"]]).astype(np.float64)
    normal_norm = np.linalg.norm(normal, axis=1)
    nz_abs = np.abs(normal[:, 2])
    return {
        "path": PATHS.relative(path),
        "file_bytes": int(path.stat().st_size),
        "header": header,
        "vertex_count": int(vertex_count),
        "xyz_min": xyz.min(axis=0).tolist(),
        "xyz_max": xyz.max(axis=0).tolist(),
        "xyz_mean": xyz.mean(axis=0).tolist(),
        "normal_norm": finite_summary(normal_norm),
        "normal_orientation_fraction": {
            "near_vertical_surface_abs_nz_lt_0.25": float(np.mean(nz_abs < 0.25)),
            "slanted_abs_nz_0.25_0.75": float(np.mean((nz_abs >= 0.25) & (nz_abs <= 0.75))),
            "near_horizontal_surface_abs_nz_gt_0.75": float(np.mean(nz_abs > 0.75)),
        },
    }


def position_stats(train_pos: np.ndarray, test_pos: np.ndarray) -> dict[str, object]:
    tree_train = cKDTree(train_pos[:, :2])
    train_nn, _ = tree_train.query(train_pos[:, :2], k=2)
    test_nn8, _ = tree_train.query(test_pos[:, :2], k=8)
    all_pos = np.vstack([train_pos, test_pos])
    return {
        "train_shape": list(train_pos.shape),
        "test_shape": list(test_pos.shape),
        "all_position_min": all_pos.min(axis=0).tolist(),
        "all_position_max": all_pos.max(axis=0).tolist(),
        "train_xyz": {axis: finite_summary(train_pos[:, i]) for i, axis in enumerate("xyz")},
        "test_xyz": {axis: finite_summary(test_pos[:, i]) for i, axis in enumerate("xyz")},
        "train_distance_to_bs": finite_summary(np.linalg.norm(train_pos - BS_POS[None, :], axis=1)),
        "test_distance_to_bs": finite_summary(np.linalg.norm(test_pos - BS_POS[None, :], axis=1)),
        "train_nearest_train_xy_distance": finite_summary(train_nn[:, 1]),
        "test_nearest_train_xy_distance": finite_summary(test_nn8[:, 0]),
        "test_eighth_train_xy_distance": finite_summary(test_nn8[:, -1]),
        "duplicate_train_positions": int(len(train_pos) - len(np.unique(train_pos, axis=0))),
        "duplicate_test_positions": int(len(test_pos) - len(np.unique(test_pos, axis=0))),
    }


def geometry_domain_analysis(
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    blocked_train: np.ndarray,
    blocked_test: np.ndarray,
    geodesic: np.lib.npyio.NpzFile,
    pointcloud: np.lib.npyio.NpzFile,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    basic_train = np.column_stack(
        [
            train_pos[:, 0],
            train_pos[:, 1],
            train_pos[:, 2],
            np.linalg.norm(train_pos - BS_POS[None, :], axis=1),
            blocked_train,
        ]
    ).astype(np.float32)
    basic_test = np.column_stack(
        [
            test_pos[:, 0],
            test_pos[:, 1],
            test_pos[:, 2],
            np.linalg.norm(test_pos - BS_POS[None, :], axis=1),
            blocked_test,
        ]
    ).astype(np.float32)
    train = np.concatenate(
        [
            basic_train,
            geodesic["train_features"].astype(np.float32),
            pointcloud["train_features"].astype(np.float32),
        ],
        axis=1,
    )
    test = np.concatenate(
        [
            basic_test,
            geodesic["test_features"].astype(np.float32),
            pointcloud["test_features"].astype(np.float32),
        ],
        axis=1,
    )
    names = [
        "x",
        "y",
        "z",
        "bs_distance",
        "blocked",
        *[f"geodesic:{value}" for value in geodesic["feature_names"]],
        *[f"pointcloud:{i}" for i in range(pointcloud["train_features"].shape[1])],
    ]
    rows = []
    for index, name in enumerate(names):
        train_values = train[:, index]
        test_values = test[:, index]
        rows.append(
            {
                "feature": name,
                "ks": float(ks_2samp(train_values, test_values).statistic),
                "psi": psi(train_values, test_values),
                "train_mean": float(np.mean(train_values)),
                "test_mean": float(np.mean(test_values)),
                "train_std": float(np.std(train_values)),
                "mean_shift_train_std": float(
                    (np.mean(test_values) - np.mean(train_values))
                    / max(float(np.std(train_values)), 1e-6)
                ),
            }
        )
    rows.sort(key=lambda row: (row["ks"], abs(row["mean_shift_train_std"])), reverse=True)
    x = np.vstack([train, test]).astype(np.float32)
    y = np.concatenate(
        [np.zeros(len(train), dtype=np.int8), np.ones(len(test), dtype=np.int8)]
    )
    clf = ExtraTreesClassifier(
        n_estimators=300,
        max_features=0.55,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=15101,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=15101)
    proba = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
    report = {
        "domain_auc": float(roc_auc_score(y, proba)),
        "train_domain_score": finite_summary(proba[y == 0]),
        "test_domain_score": finite_summary(proba[y == 1]),
        "top_shift_features": rows[:20],
    }
    return report, rows


def channel_structure(
    support_raw: np.ndarray,
    valid: np.ndarray,
    regimes_train: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, np.ndarray]]:
    total_power = support_raw.sum(axis=(1, 2, 3))
    support = support_raw[valid].astype(np.float32)
    support /= np.maximum(support.sum(axis=(1, 2, 3), keepdims=True), TINY)
    h = support.sum(axis=(2, 3))
    v = support.sum(axis=(1, 3))
    d = support.sum(axis=(1, 2))
    flat = support.reshape(len(support), -1)
    sorted_flat = np.sort(flat, axis=1)[:, ::-1]
    top1 = sorted_flat[:, 0]
    top8 = sorted_flat[:, :8].sum(axis=1)
    top64 = sorted_flat[:, :64].sum(axis=1)
    h_entropy = entropy01(h, axis=1)
    v_entropy = entropy01(v, axis=1)
    d_entropy = entropy01(d, axis=1)
    joint_entropy = entropy01(flat, axis=1)
    delay_bins = np.arange(d.shape[1], dtype=np.float32)
    delay_mean = (d * delay_bins[None, :]).sum(axis=1)
    delay_rms = np.sqrt(
        np.maximum((d * (delay_bins[None, :] - delay_mean[:, None]) ** 2).sum(axis=1), 0.0)
    )
    h_angles = np.exp(2j * np.pi * np.arange(h.shape[1]) / h.shape[1])
    h_resultant = np.abs((h * h_angles[None, :]).sum(axis=1))
    h_circular_spread = 1.0 - h_resultant
    metrics = {
        "total_power": total_power,
        "log10_total_power": np.log10(np.maximum(total_power, TINY)),
        "h_entropy": h_entropy,
        "v_entropy": v_entropy,
        "delay_entropy": d_entropy,
        "joint_entropy": joint_entropy,
        "joint_top1": top1,
        "joint_top8": top8,
        "joint_top64": top64,
        "delay_mean_bin": delay_mean,
        "delay_rms_spread": delay_rms,
        "h_circular_spread": h_circular_spread.astype(np.float32),
    }
    overall = {
        "train_points": int(len(support_raw)),
        "valid_nonzero_points": int(np.sum(valid)),
        "exact_zero_channel_points": int(np.sum(~valid)),
        "near_zero_power_lt_1e-20": int(np.sum(total_power < 1e-20)),
        "total_power": finite_summary(total_power),
        "log10_total_power_valid": finite_summary(metrics["log10_total_power"][valid]),
        "h_entropy": finite_summary(h_entropy),
        "v_entropy": finite_summary(v_entropy),
        "delay_entropy": finite_summary(d_entropy),
        "joint_entropy": finite_summary(joint_entropy),
        "joint_top1": finite_summary(top1),
        "joint_top8": finite_summary(top8),
        "joint_top64": finite_summary(top64),
        "delay_mean_bin": finite_summary(delay_mean),
        "delay_rms_spread": finite_summary(delay_rms),
        "h_circular_spread": finite_summary(h_circular_spread),
    }
    valid_regimes = regimes_train[valid]
    rows = []
    for regime in REGIMES:
        mask = valid_regimes == regime
        row = {
            "regime": regime,
            "count": int(np.sum(mask)),
            "zero_count": int(np.sum((regimes_train == regime) & (~valid))),
        }
        for name in (
            "log10_total_power",
            "h_entropy",
            "v_entropy",
            "delay_entropy",
            "joint_entropy",
            "joint_top1",
            "joint_top8",
            "delay_mean_bin",
            "delay_rms_spread",
            "h_circular_spread",
        ):
            values = metrics[name][valid][mask] if name == "log10_total_power" else metrics[name][mask]
            row[f"{name}_mean"] = float(np.mean(values)) if np.any(mask) else 0.0
            row[f"{name}_median"] = float(np.median(values)) if np.any(mask) else 0.0
        rows.append(row)
    return overall, rows, {"h": h, "v": v, "delay": d, "metrics": metrics}


def spatial_correlations(
    train_pos: np.ndarray,
    valid: np.ndarray,
    regimes_train: np.ndarray,
    support_parts: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    xy = train_pos[valid, :2].astype(np.float32)
    dist = pdist(xy, metric="euclidean")
    bins = np.asarray([0, 5, 10, 15, 20, 30, 40, 60, 80, 120, np.inf], dtype=np.float32)
    bin_index = np.digitize(dist, bins[1:-1], right=False)
    rows = []
    scalar_metrics = support_parts["metrics"]
    selected_scalar = {
        "log10_total_power": scalar_metrics["log10_total_power"][valid],
        "h_entropy": scalar_metrics["h_entropy"],
        "delay_entropy": scalar_metrics["delay_entropy"],
        "joint_top1": scalar_metrics["joint_top1"],
        "delay_mean_bin": scalar_metrics["delay_mean_bin"],
        "delay_rms_spread": scalar_metrics["delay_rms_spread"],
        "h_circular_spread": scalar_metrics["h_circular_spread"],
    }
    scalar_semivar = {}
    for name, values in selected_scalar.items():
        z = (values - np.mean(values)) / max(float(np.std(values)), 1e-6)
        scalar_semivar[name] = 0.5 * pdist(z[:, None], metric="sqeuclidean")
    h_cos = 1.0 - pdist(support_parts["h"], metric="cosine")
    v_cos = 1.0 - pdist(support_parts["v"], metric="cosine")
    d_cos = 1.0 - pdist(support_parts["delay"], metric="cosine")
    same_regime = pdist(regimes_train[valid, None], metric=lambda a, b: float(a[0] != b[0]))
    for i in range(len(bins) - 1):
        mask = bin_index == i
        if not np.any(mask):
            continue
        row = {
            "distance_low": float(bins[i]),
            "distance_high": (
                None if not np.isfinite(bins[i + 1]) else float(bins[i + 1])
            ),
            "pair_count": int(np.sum(mask)),
            "same_regime_fraction": float(np.mean(same_regime[mask] == 0.0)),
            "h_marginal_cosine": float(np.mean(h_cos[mask])),
            "v_marginal_cosine": float(np.mean(v_cos[mask])),
            "delay_marginal_cosine": float(np.mean(d_cos[mask])),
        }
        for name, semivar in scalar_semivar.items():
            row[f"{name}_semivariance"] = float(np.mean(semivar[mask]))
        rows.append(row)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup = json.loads((PATHS.round1 / "Round1_Setup.json").read_text())
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    train_channel = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    support_raw = np.load(PATHS.shared_features / "truth_coll_power.npy", mmap_mode="r")
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked_train = height["train_blocked"].astype(np.float32)
    blocked_test = height["test_blocked"].astype(np.float32)
    regimes_train = regime_of(blocked_train)
    regimes_test = regime_of(blocked_test)
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    pointcloud = np.load(PATHS.shared_features / "official_pointcloud_scene_cache.npz")
    total_power = np.asarray(support_raw).sum(axis=(1, 2, 3))
    valid = total_power > 0

    geometry_report, geometry_rows = geometry_domain_analysis(
        train_pos, test_pos, blocked_train, blocked_test, geodesic, pointcloud
    )
    channel_report, regime_rows, support_parts = channel_structure(
        np.asarray(support_raw), valid, regimes_train
    )
    spatial_rows = spatial_correlations(train_pos, valid, regimes_train, support_parts)
    with GEOMETRY_SHIFT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(geometry_rows[0].keys()))
        writer.writeheader()
        writer.writerows(geometry_rows)
    with REGIME_CHANNEL_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(regime_rows[0].keys()))
        writer.writeheader()
        writer.writerows(regime_rows)
    with SPATIAL_CORR_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(spatial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(spatial_rows)

    report = {
        "method": "official_data_multidim_analysis",
        "status": "analysis_only_no_candidate",
        "official_files": {
            "setup": setup,
            "train_pos_shape": list(train_pos.shape),
            "test_pos_shape": list(test_pos.shape),
            "train_channel_shape": list(train_channel.shape),
            "train_channel_dtype": str(train_channel.dtype),
            "setup_p_train_note": (
                "Round1_Setup P_Train is 2500, while actual Train_Pos/Channel "
                "contain 2000 and Test_Pos contains 500; this appears to mean "
                "total provided points rather than train-only points."
            ),
        },
        "position": position_stats(train_pos, test_pos),
        "pointcloud_map": pointcloud_stats(),
        "geometry": {
            "train_regime_counts": {
                regime: int(np.sum(regimes_train == regime)) for regime in REGIMES
            },
            "test_regime_counts": {
                regime: int(np.sum(regimes_test == regime)) for regime in REGIMES
            },
            "blocked_train": finite_summary(blocked_train),
            "blocked_test": finite_summary(blocked_test),
            "geodesic_feature_names": [str(value) for value in geodesic["feature_names"]],
            "domain_analysis": geometry_report,
            "top_shift_csv": PATHS.relative(GEOMETRY_SHIFT_CSV),
        },
        "channel_train_truth": channel_report,
        "regime_channel_csv": PATHS.relative(REGIME_CHANNEL_CSV),
        "spatial_correlation_csv": PATHS.relative(SPATIAL_CORR_CSV),
        "spatial_correlation_summary": spatial_rows,
        "initial_interpretation": {
            "geometry_split_is_balanced": geometry_report["domain_auc"] < 0.60,
            "channel_has_exact_zero_outliers": channel_report["exact_zero_channel_points"] > 0,
            "ad_support_is_sparse": channel_report["joint_top8"]["median"] > 0.5,
            "spatial_channel_structure_available": (
                spatial_rows[0]["h_marginal_cosine"] > spatial_rows[-1]["h_marginal_cosine"]
            ),
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "geometry_shift_csv": PATHS.relative(GEOMETRY_SHIFT_CSV),
            "regime_channel_csv": PATHS.relative(REGIME_CHANNEL_CSV),
            "spatial_correlation_csv": PATHS.relative(SPATIAL_CORR_CSV),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
