#!/usr/bin/env python3
"""Probe confocal point-cloud arrival regions for official normalized PDP D."""

from __future__ import annotations

import json
import math
import time

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

import analyze_v274_planar_barrier_attention as v274
import analyze_v279_joint_component_trajectory as v279
import analyze_v283_joint_geometry_path_renderer as v283
import analyze_v315_full_sequence_confirmatory as v315
from project_paths import PATHS


VERSION_NUMBER = 393
VERSION = "v393_confocal_pointcloud_pdp"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
FEATURE_CACHE = OUT_DIR / "v393_confocal_pointcloud_features.npz"
OOF_CACHE = OUT_DIR / "v393_confocal_pointcloud_pdp_oof.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"

LIGHT_SPEED = 299_792_458.0
SUBCARRIER_SPACING = 30_000.0
DELAY_BINS = 192
ACTIVE_BINS = 24
SUBSHELLS = 4
BIN_LENGTH = LIGHT_SPEED / (DELAY_BINS * SUBCARRIER_SPACING)
SUBSHELL_LENGTH = BIN_LENGTH / SUBSHELLS
TREE_COUNT = 160
MAX_DEPTH = 22
MIN_LEAF = 8
MAX_FEATURES = 0.65
ALPHAS = (0.0, 0.10, 0.25, 0.50, 1.0)
EPS = 2e-5

MIN_COSINE_LIFT = 0.002
MIN_FULL_VS_CONTEXT = 0.001
MIN_FULL_VS_SHUFFLED = 0.001
MIN_EMD_REDUCTION = 0.05
MIN_FOLD_WINS = 4

FEATURE_NAMES = (
    "log_cell_count",
    "log_vertical_point_count",
    "mean_height",
    "max_height",
    "mean_specular_alignment",
    "max_specular_alignment",
    "specular_fraction_080",
    "specular_fraction_095",
    "mean_incidence_product",
    "mean_bs_path_fraction",
    "std_bs_path_fraction",
    "normal_concentration",
    "mean_transverse_distance",
)


def normalize(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    value /= np.maximum(value.sum(axis=1, keepdims=True), 1e-30)
    return value


def row_cos(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return v279.row_cos(truth, prediction).astype(np.float64)


def row_emd(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.mean(
        np.abs(np.cumsum(truth, axis=1) - np.cumsum(prediction, axis=1)),
        axis=1,
    ).astype(np.float64)


def wall_cells() -> dict[str, np.ndarray]:
    scene = np.load(SCENE_CACHE)
    vertical = scene["grid_vertical_count"].astype(np.float32)
    z_min = scene["grid_z_min"].astype(np.float32)
    z_max = scene["grid_z_max"].astype(np.float32)
    z_mean = scene["grid_z_mean"].astype(np.float32)
    nx = scene["grid_wall_nx"].astype(np.float32)
    ny = scene["grid_wall_ny"].astype(np.float32)
    keep = (vertical > 0) & np.isfinite(z_max) & (z_max > 1.5)
    iy, ix = np.nonzero(keep)
    resolution = float(scene["resolution"])
    x = float(scene["x_min"]) + (ix.astype(np.float32) + 0.5) * resolution
    y = float(scene["y_min"]) + (iy.astype(np.float32) + 0.5) * resolution
    low = np.maximum(z_min[keep], 1.5)
    high = np.maximum(z_max[keep], low)
    z = np.clip(z_mean[keep], low, high)
    bad_z = ~np.isfinite(z)
    z[bad_z] = 0.5 * (low[bad_z] + high[bad_z])
    normal = np.column_stack([nx[keep], ny[keep], np.zeros(len(x))]).astype(
        np.float32
    )
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    weight = np.maximum(vertical[keep], 1.0).astype(np.float32)
    return {
        "point": np.column_stack([x, y, z]).astype(np.float32),
        "normal": normal,
        "weight": weight,
        "height": high.astype(np.float32),
    }


def grouped_sum(index: np.ndarray, value: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(index, weights=value, minlength=size).astype(np.float32)


def build_confocal_features(positions: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    cells = wall_cells()
    point = cells["point"].astype(np.float64)
    normal = cells["normal"].astype(np.float64)
    weight = cells["weight"].astype(np.float64)
    height = cells["height"].astype(np.float64)
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    to_bs = bs[None, :] - point
    distance_bs = np.linalg.norm(to_bs, axis=1)
    unit_bs = to_bs / np.maximum(distance_bs[:, None], 1e-8)
    fine_count = ACTIVE_BINS * SUBSHELLS
    output = np.zeros(
        (len(positions), ACTIVE_BINS, SUBSHELLS, len(FEATURE_NAMES)),
        dtype=np.float32,
    )
    started = time.time()
    for row, ue_raw in enumerate(positions):
        ue = np.asarray(ue_raw, dtype=np.float64)
        to_ue = ue[None, :] - point
        distance_ue = np.linalg.norm(to_ue, axis=1)
        unit_ue = to_ue / np.maximum(distance_ue[:, None], 1e-8)
        total = distance_bs + distance_ue
        fine_bin = np.floor(total / SUBSHELL_LENGTH).astype(np.int64)
        keep = (fine_bin >= 0) & (fine_bin < fine_count)
        if not np.any(keep):
            continue
        index = fine_bin[keep]
        current_weight = weight[keep]
        count = np.bincount(index, minlength=fine_count).astype(np.float32)
        mass = grouped_sum(index, current_weight, fine_count)

        reflection_normal = unit_bs[keep] + unit_ue[keep]
        reflection_normal /= np.maximum(
            np.linalg.norm(reflection_normal, axis=1, keepdims=True), 1e-8
        )
        alignment = np.abs(np.sum(normal[keep] * reflection_normal, axis=1))
        incidence = np.abs(np.sum(normal[keep] * unit_bs[keep], axis=1))
        incidence *= np.abs(np.sum(normal[keep] * unit_ue[keep], axis=1))
        path_fraction = distance_bs[keep] / np.maximum(total[keep], 1e-8)

        segment = ue - bs
        segment_norm2 = max(float(np.dot(segment, segment)), 1e-8)
        relative = point[keep] - bs[None, :]
        projection = np.clip(relative @ segment / segment_norm2, 0.0, 1.0)
        closest = bs[None, :] + projection[:, None] * segment[None, :]
        transverse = np.linalg.norm(point[keep] - closest, axis=1)

        feature = np.zeros((fine_count, len(FEATURE_NAMES)), dtype=np.float32)
        feature[:, 0] = np.log1p(count)
        feature[:, 1] = np.log1p(mass)
        feature[:, 2] = grouped_sum(
            index, current_weight * height[keep], fine_count
        ) / np.maximum(mass, 1e-8) / 40.0
        maximum_height = np.zeros(fine_count, dtype=np.float64)
        np.maximum.at(maximum_height, index, height[keep])
        feature[:, 3] = (maximum_height / 40.0).astype(np.float32)
        feature[:, 4] = grouped_sum(
            index, current_weight * alignment, fine_count
        ) / np.maximum(mass, 1e-8)
        maximum_alignment = np.zeros(fine_count, dtype=np.float64)
        np.maximum.at(maximum_alignment, index, alignment)
        feature[:, 5] = maximum_alignment.astype(np.float32)
        feature[:, 6] = grouped_sum(
            index, current_weight * (alignment >= 0.80), fine_count
        ) / np.maximum(mass, 1e-8)
        feature[:, 7] = grouped_sum(
            index, current_weight * (alignment >= 0.95), fine_count
        ) / np.maximum(mass, 1e-8)
        feature[:, 8] = grouped_sum(
            index, current_weight * incidence, fine_count
        ) / np.maximum(mass, 1e-8)
        mean_fraction = grouped_sum(
            index, current_weight * path_fraction, fine_count
        ) / np.maximum(mass, 1e-8)
        mean_fraction2 = grouped_sum(
            index, current_weight * path_fraction**2, fine_count
        ) / np.maximum(mass, 1e-8)
        feature[:, 9] = mean_fraction
        feature[:, 10] = np.sqrt(
            np.maximum(mean_fraction2 - mean_fraction**2, 0.0)
        )
        normal_x = grouped_sum(
            index, current_weight * normal[keep, 0], fine_count
        )
        normal_y = grouped_sum(
            index, current_weight * normal[keep, 1], fine_count
        )
        feature[:, 11] = np.sqrt(normal_x**2 + normal_y**2) / np.maximum(mass, 1e-8)
        feature[:, 12] = grouped_sum(
            index, current_weight * transverse, fine_count
        ) / np.maximum(mass, 1e-8) / 100.0
        output[row] = feature.reshape(
            ACTIVE_BINS, SUBSHELLS, len(FEATURE_NAMES)
        )
        if (row + 1) % 250 == 0 or row + 1 == len(positions):
            print(
                f"v393 confocal features {row + 1}/{len(positions)} "
                f"seconds={time.time() - started:.1f}",
                flush=True,
            )
    meta = {
        "wall_cell_count": int(len(point)),
        "feature_shape": list(output.shape),
        "seconds": float(time.time() - started),
        "bin_length_m": BIN_LENGTH,
        "subshell_length_m": SUBSHELL_LENGTH,
    }
    return output, meta


def load_or_build_features() -> tuple[np.ndarray, dict[str, object]]:
    if FEATURE_CACHE.is_file():
        cached = np.load(FEATURE_CACHE)
        return cached["feature"].astype(np.float32), json.loads(str(cached["meta"]))
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    feature, meta = build_confocal_features(np.vstack([train_pos, test_pos]))
    np.savez_compressed(FEATURE_CACHE, feature=feature, meta=np.asarray(json.dumps(meta)))
    return feature, meta


def distance_stratified_permutation(
    positions: np.ndarray, seed: int, strata: int = 20
) -> np.ndarray:
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    distance = np.linalg.norm(positions - bs[None, :], axis=1)
    edges = np.quantile(distance, np.linspace(0.0, 1.0, strata + 1))
    group = np.clip(np.searchsorted(edges[1:-1], distance, side="right"), 0, strata - 1)
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(positions), dtype=np.int64)
    for value in range(strata):
        rows = np.flatnonzero(group == value)
        permutation[rows] = rng.permutation(rows)
    return permutation


def anchor_d(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    positions: np.ndarray,
    truth_sh: np.ndarray,
) -> np.ndarray:
    sh = v283.knn_anchor(query_ids, fit_ids, positions, truth_sh)
    return normalize(v279.derive_d_numpy(sh))


def bin_context(anchor: np.ndarray, positions: np.ndarray) -> np.ndarray:
    count = len(anchor)
    index = np.arange(ACTIVE_BINS, dtype=np.float32)
    angle = 2.0 * math.pi * index / DELAY_BINS
    bin_feature = np.column_stack(
        [
            index / ACTIVE_BINS,
            np.sin(angle),
            np.cos(angle),
            np.sin(2.0 * angle),
            np.cos(2.0 * angle),
        ]
    ).astype(np.float32)
    active = anchor[:, :ACTIVE_BINS]
    log_active = np.log(np.maximum(active, EPS))
    local = np.stack(
        [
            active,
            np.roll(active, 1, axis=1),
            np.roll(active, -1, axis=1),
            np.roll(active, 2, axis=1),
            np.roll(active, -2, axis=1),
            log_active,
            np.cumsum(active, axis=1),
        ],
        axis=2,
    )
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    direct = np.linalg.norm(positions - bs[None, :], axis=1) / BIN_LENGTH
    direct_feature = np.stack(
        [
            (index[None, :] >= np.floor(direct)[:, None]).astype(np.float32),
            np.broadcast_to((direct / ACTIVE_BINS)[:, None], (count, ACTIVE_BINS)),
            np.broadcast_to(
                (np.argmax(anchor, axis=1) / DELAY_BINS)[:, None],
                (count, ACTIVE_BINS),
            ),
        ],
        axis=2,
    )
    return np.concatenate(
        [
            local,
            np.broadcast_to(bin_feature[None, :, :], (count,) + bin_feature.shape),
            direct_feature,
        ],
        axis=2,
    ).astype(np.float32)


def model_matrix(
    ids: np.ndarray,
    anchor: np.ndarray,
    positions: np.ndarray,
    scene_feature: np.ndarray,
    confocal: np.ndarray,
    fit_ids: np.ndarray,
    mode: str,
    permutation: np.ndarray,
) -> np.ndarray:
    global_scaler = StandardScaler().fit(scene_feature[fit_ids])
    global_value = global_scaler.transform(scene_feature[ids]).astype(np.float32)
    context = bin_context(anchor, positions[ids])
    if mode == "context":
        physical = np.zeros(
            (len(ids), ACTIVE_BINS, SUBSHELLS * len(FEATURE_NAMES)),
            dtype=np.float32,
        )
    else:
        physical_ids = ids if mode == "full" else permutation[ids]
        physical = confocal[physical_ids].reshape(len(ids), ACTIVE_BINS, -1)
        fit_physical_ids = fit_ids if mode == "full" else permutation[fit_ids]
        physical_scaler = StandardScaler().fit(
            confocal[fit_physical_ids].reshape(-1, physical.shape[2])
        )
        physical = physical_scaler.transform(
            physical.reshape(-1, physical.shape[2])
        ).reshape(physical.shape).astype(np.float32)
    global_repeated = np.broadcast_to(
        global_value[:, None, :], (len(ids), ACTIVE_BINS, global_value.shape[1])
    )
    return np.concatenate([context, global_repeated, physical], axis=2).reshape(
        -1, context.shape[2] + global_value.shape[1] + physical.shape[2]
    )


def fit_predict_residual(
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    fit_anchor: np.ndarray,
    query_anchor: np.ndarray,
    truth_d: np.ndarray,
    positions: np.ndarray,
    scene_feature: np.ndarray,
    confocal: np.ndarray,
    mode: str,
    permutation: np.ndarray,
    seed: int,
) -> np.ndarray:
    x_fit = model_matrix(
        fit_ids, fit_anchor, positions, scene_feature, confocal,
        fit_ids, mode, permutation,
    )
    x_query = model_matrix(
        query_ids, query_anchor, positions, scene_feature, confocal,
        fit_ids, mode, permutation,
    )
    target = np.log(
        (truth_d[fit_ids, :ACTIVE_BINS] + EPS)
        / (fit_anchor[:, :ACTIVE_BINS] + EPS)
    )
    target = np.clip(target, -4.0, 4.0).reshape(-1)
    importance = 0.10 + 10.0 * (
        truth_d[fit_ids, :ACTIVE_BINS] + fit_anchor[:, :ACTIVE_BINS]
    )
    importance = np.clip(importance, 0.10, 5.0).reshape(-1)
    model = ExtraTreesRegressor(
        n_estimators=TREE_COUNT,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_LEAF,
        max_features=MAX_FEATURES,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x_fit, target, sample_weight=importance)
    return np.clip(
        model.predict(x_query).reshape(len(query_ids), ACTIVE_BINS), -4.0, 4.0
    ).astype(np.float32)


def apply_residual(anchor: np.ndarray, residual: np.ndarray, alpha: float) -> np.ndarray:
    prediction = anchor.copy()
    prediction[:, :ACTIVE_BINS] *= np.exp(alpha * residual).astype(np.float32)
    return normalize(prediction)


def choose_alpha(
    truth: np.ndarray, anchor: np.ndarray, residual: np.ndarray
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for alpha in ALPHAS:
        prediction = apply_residual(anchor, residual, alpha)
        rows.append(
            {
                "alpha": alpha,
                "cosine": float(np.mean(row_cos(truth, prediction))),
                "emd": float(np.mean(row_emd(truth, prediction))),
            }
        )
    selected = max(rows, key=lambda row: (row["cosine"], -row["emd"]))
    return float(selected["alpha"]), rows


def main() -> None:
    for path in (TARGET_CACHE, SCENE_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    confocal_all, feature_meta = load_or_build_features()
    target = np.load(TARGET_CACHE)
    truth_sh = v279.normalize(target["sh"].astype(np.float32))
    truth_d = normalize(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    all_pos = np.vstack([train_pos, test_pos])
    scene = np.load(SCENE_CACHE)
    scene_feature = np.vstack(
        [scene["train_features"], scene["test_features"]]
    ).astype(np.float32)
    permutation = distance_stratified_permutation(all_pos, 39302)
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    modes = ("context", "shuffled", "full")
    fold_rows = []
    oof: dict[str, list[np.ndarray]] = {
        key: [] for key in ("ids", "truth", "anchor", *modes)
    }
    for outer_fold, outer_ids in enumerate(folds):
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = folds[(outer_fold + 1) % len(folds)]
        inner_fit = np.setdiff1d(outer_fit, inner_ids)
        inner_fit_anchor = anchor_d(inner_fit, inner_fit, train_pos, truth_sh)
        inner_val_anchor = anchor_d(inner_ids, inner_fit, train_pos, truth_sh)
        outer_fit_anchor = anchor_d(outer_fit, outer_fit, train_pos, truth_sh)
        outer_anchor = anchor_d(outer_ids, outer_fit, train_pos, truth_sh)
        predictions = {}
        mode_reports = {}
        for mode_index, mode in enumerate(modes):
            inner_residual = fit_predict_residual(
                inner_fit, inner_ids, inner_fit_anchor, inner_val_anchor,
                truth_d, all_pos, scene_feature, confocal_all,
                mode, permutation, 393_000 + 100 * outer_fold + mode_index,
            )
            alpha, selection = choose_alpha(
                truth_d[inner_ids], inner_val_anchor, inner_residual
            )
            outer_residual = fit_predict_residual(
                outer_fit, outer_ids, outer_fit_anchor, outer_anchor,
                truth_d, all_pos, scene_feature, confocal_all,
                mode, permutation, 393_500 + 100 * outer_fold + mode_index,
            )
            prediction = apply_residual(outer_anchor, outer_residual, alpha)
            predictions[mode] = prediction
            mode_reports[mode] = {
                "alpha": alpha,
                "selection": selection,
                "cosine": float(np.mean(row_cos(truth_d[outer_ids], prediction))),
                "emd": float(np.mean(row_emd(truth_d[outer_ids], prediction))),
            }
        anchor_cosine = float(np.mean(row_cos(truth_d[outer_ids], outer_anchor)))
        anchor_emd = float(np.mean(row_emd(truth_d[outer_ids], outer_anchor)))
        delta = {
            "full_vs_anchor_cosine": mode_reports["full"]["cosine"] - anchor_cosine,
            "full_vs_context_cosine": mode_reports["full"]["cosine"] - mode_reports["context"]["cosine"],
            "full_vs_shuffled_cosine": mode_reports["full"]["cosine"] - mode_reports["shuffled"]["cosine"],
            "full_emd_reduction": (anchor_emd - mode_reports["full"]["emd"]) / max(anchor_emd, 1e-30),
        }
        fold_rows.append(
            {
                "fold": outer_fold,
                "query_count": int(len(outer_ids)),
                "inner_query_count": int(len(inner_ids)),
                "anchor": {"cosine": anchor_cosine, "emd": anchor_emd},
                "modes": mode_reports,
                "deltas": delta,
            }
        )
        oof["ids"].append(outer_ids)
        oof["truth"].append(truth_d[outer_ids])
        oof["anchor"].append(outer_anchor)
        for mode in modes:
            oof[mode].append(predictions[mode])
        print(
            f"v393 fold={outer_fold} alpha={mode_reports['full']['alpha']:.2f} "
            f"delta={delta['full_vs_anchor_cosine']:+.6f} "
            f"vs_context={delta['full_vs_context_cosine']:+.6f} "
            f"vs_shuffle={delta['full_vs_shuffled_cosine']:+.6f}",
            flush=True,
        )
    merged = {key: np.concatenate(value) for key, value in oof.items()}
    order = np.argsort(merged["ids"])
    merged = {key: value[order] for key, value in merged.items()}
    np.savez_compressed(OOF_CACHE, **merged)
    summary = {
        key: float(np.mean([row["deltas"][key] for row in fold_rows]))
        for key in fold_rows[0]["deltas"]
    }
    summary["fold_wins"] = int(
        sum(
            row["deltas"]["full_vs_anchor_cosine"] > 0
            and row["deltas"]["full_vs_context_cosine"] > 0
            and row["deltas"]["full_vs_shuffled_cosine"] > 0
            for row in fold_rows
        )
    )
    passed = bool(
        summary["full_vs_anchor_cosine"] >= MIN_COSINE_LIFT
        and summary["full_vs_context_cosine"] >= MIN_FULL_VS_CONTEXT
        and summary["full_vs_shuffled_cosine"] >= MIN_FULL_VS_SHUFFLED
        and summary["full_emd_reduction"] >= MIN_EMD_REDUCTION
        and summary["fold_wins"] >= MIN_FOLD_WINS
    )
    report = {
        "method": VERSION,
        "status": "confocal_feature_passed" if passed else "analysis_only_no_transfer",
        "primary_hypothesis": (
            "Delay-conditioned co-focal point-cloud arrival regions contain path lifecycle "
            "information beyond a strong official KNN D anchor and point-level scene context."
        ),
        "literature_basis": {
            "title": "Point Cloud Environment-Based Channel Knowledge Map Construction",
            "arxiv": "2506.21112",
            "adaptation": (
                "absolute-delay ellipsoidal regions, four physical subshells per official "
                "delay bin, official vertical point-cloud cells and surface-normal features"
            ),
        },
        "parameters": {
            "active_bins": ACTIVE_BINS,
            "subshells": SUBSHELLS,
            "bin_length_m": BIN_LENGTH,
            "subshell_length_m": SUBSHELL_LENGTH,
            "feature_names": list(FEATURE_NAMES),
            "tree_count": TREE_COUNT,
            "max_depth": MAX_DEPTH,
            "min_leaf": MIN_LEAF,
            "max_features": MAX_FEATURES,
            "alphas": list(ALPHAS),
            "folds": "five mutually exclusive isolation-stratified folds",
            "inner_selection": "next disjoint fold; outer truth never selects alpha",
        },
        "feature_cache": feature_meta,
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "full_vs_anchor_cosine": f">={MIN_COSINE_LIFT}",
            "full_vs_context_cosine": f">={MIN_FULL_VS_CONTEXT}",
            "full_vs_shuffled_cosine": f">={MIN_FULL_VS_SHUFFLED}",
            "full_emd_reduction": f">={MIN_EMD_REDUCTION}",
            "fold_wins": f">={MIN_FOLD_WINS}/5",
            "passed": passed,
        },
        "compliance_boundary": {
            "uses": [
                "official train H normalized D labels",
                "official train/test coordinates",
                "official Round1_Map point cloud and normals",
                "an AI residual regressor with nested official-only selection",
            ],
            "does_not_use": [
                "ray-traced channel targets or path coefficients",
                "restored mesh/material/object labels",
                "v212 or synthetic teacher outputs",
                "hidden/online truth",
            ],
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "feature_cache": PATHS.relative(FEATURE_CACHE),
            "oof_cache": PATHS.relative(OOF_CACHE),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(report["decision_rule"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
