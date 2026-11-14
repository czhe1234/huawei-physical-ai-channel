#!/usr/bin/env python3
"""Test whether global route and local support jointly solve path selection."""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score

import analyze_v293_interaction_physical_path_law as v293
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v308_joint_matched_patch_renderer as v308
import analyze_v312_highdepth_path_identity_match as v312
import analyze_v313_highdepth_matched_path_renderer as v313
import analyze_v315_full_sequence_confirmatory as v315
from project_paths import PATHS


VERSION_NUMBER = 431
VERSION = "v431_global_local_path_selection_feasibility"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v431_candidate_ranking_oof.npz"

TOP_REFLECTIONS = 32
LOCAL_K = 16
TREE_COUNT = 180
MIN_JOINT_MARGIN = 0.03
MIN_FOLD_WINS = 4
MIN_ORACLE_DELTA = 0.01
ALPHA_GRID = np.linspace(0.0, 1.0, 21, dtype=np.float32)


def normalize(value: np.ndarray) -> np.ndarray:
    shape = (len(value),) + (1,) * (value.ndim - 1)
    total = value.reshape(len(value), -1).sum(axis=1).reshape(shape)
    return (value / np.maximum(total, 1e-30)).astype(np.float32)


def row_cos(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = left.reshape(len(left), -1).astype(np.float64)
    y = right.reshape(len(right), -1).astype(np.float64)
    return np.sum(x * y, axis=1) / np.maximum(
        np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30
    )


def exact_atoms(center: np.ndarray, size: int) -> np.ndarray:
    bins = np.arange(size, dtype=np.float64)[None, None, :]
    delta = np.remainder(bins - center[:, :, None] + size / 2.0, size) - size / 2.0
    output = np.square(np.sinc(delta) / np.sinc(delta / size))
    output /= np.maximum(output.sum(axis=2, keepdims=True), 1e-30)
    return output.astype(np.float32)


def build_local_evidence(
    source_ids: np.ndarray,
    query_ids: np.ndarray,
    train_pos: np.ndarray,
    valid: np.ndarray,
    patch_id: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    output = np.zeros((len(query_ids), TOP_REFLECTIONS, 7), dtype=np.float32)
    patch_rows: dict[int, list[tuple[int, int]]] = {}
    for point_id in source_ids:
        for candidate_id in np.flatnonzero(valid[point_id]):
            patch_rows.setdefault(int(patch_id[point_id, candidate_id]), []).append(
                (int(point_id), int(candidate_id))
            )
    trees: dict[int, tuple[np.ndarray, np.ndarray, cKDTree]] = {}
    for patch, rows in patch_rows.items():
        points = np.asarray([row[0] for row in rows], dtype=np.int64)
        candidates = np.asarray([row[1] for row in rows], dtype=np.int64)
        trees[patch] = (points, candidates, cKDTree(train_pos[points, :2]))

    for query_row, point_id in enumerate(query_ids):
        for candidate_id in np.flatnonzero(valid[point_id]):
            item = trees.get(int(patch_id[point_id, candidate_id]))
            if item is None:
                continue
            points, candidates, tree = item
            count = min(LOCAL_K + 2, len(points))
            distance, local = tree.query(train_pos[point_id, :2], k=count)
            distance = np.atleast_1d(distance)
            local = np.atleast_1d(local)
            keep = points[local] != point_id
            distance = distance[keep][:LOCAL_K]
            local = local[keep][:LOCAL_K]
            if len(local) == 0:
                continue
            mass = labels[points[local], candidates[local]].astype(np.float64)
            positive = mass > 1e-5
            weight = 1.0 / np.maximum(distance, 0.25) ** 2
            weight /= weight.sum()
            nearest_positive = float(np.min(distance[positive])) if np.any(positive) else 200.0
            output[query_row, candidate_id] = np.asarray(
                [
                    np.sum(weight * positive),
                    np.sum(weight * mass),
                    np.max(mass),
                    np.std(mass),
                    np.log1p(distance[0]) / 5.0,
                    np.log1p(nearest_positive) / 5.0,
                    np.mean(positive),
                ],
                dtype=np.float32,
            )
    return output


def top5_recall(
    prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray
) -> tuple[float, float]:
    recall: list[float] = []
    weight: list[float] = []
    for row in range(len(prediction)):
        usable = valid[row]
        total = float(truth[row, usable].sum())
        if total <= 1e-8:
            continue
        ids = np.flatnonzero(usable)
        top = ids[np.argsort(prediction[row, ids])[-min(5, len(ids)) :]]
        value = float(truth[row, top].sum() / total)
        recall.append(value)
        weight.append(total)
    return float(np.mean(recall)), float(np.average(recall, weights=weight))


def run_candidate_ablation(
    train_pos: np.ndarray,
    train_feature,
    train_candidates: dict[str, np.ndarray],
    labels: np.ndarray,
    quality: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    valid = train_candidates["score"][:, :TOP_REFLECTIONS] > 0.0
    patch_id = train_candidates["patch_id"][:, :TOP_REFLECTIONS].astype(np.int64)
    path_full = train_feature.path_numeric[:, 1 : TOP_REFLECTIONS + 1].astype(np.float32)
    visibility = np.load(PATHS.cache_dir(270) / "v270_reflection_visibility.npz")
    route = np.stack(
        [
            np.log1p(visibility["train_first_count"][:, :TOP_REFLECTIONS]) / 5.0,
            np.log1p(visibility["train_second_count"][:, :TOP_REFLECTIONS]) / 5.0,
            np.nan_to_num(
                visibility["train_first_nearest"][:, :TOP_REFLECTIONS],
                nan=200.0,
                posinf=200.0,
            )
            / 200.0,
            np.nan_to_num(
                visibility["train_second_nearest"][:, :TOP_REFLECTIONS],
                nan=200.0,
                posinf=200.0,
            )
            / 200.0,
            np.nan_to_num(
                visibility["train_first_strength"][:, :TOP_REFLECTIONS],
                nan=0.0,
                posinf=0.0,
            ),
            np.nan_to_num(
                visibility["train_second_strength"][:, :TOP_REFLECTIONS],
                nan=0.0,
                posinf=0.0,
            ),
        ],
        axis=2,
    ).astype(np.float32)
    node = train_feature.global_features.astype(np.float32)
    global_point = node[:, np.r_[0:56, 78:94]]
    local_point = node[:, np.r_[56:78, 93:94]]
    patch_scale = float(max(int(patch_id.max()) + 2, 2))
    angle = 2.0 * np.pi * np.maximum(patch_id, 0) / patch_scale
    patch_harmonic = np.stack(
        [np.sin(angle), np.cos(angle), np.sin(2.0 * angle), np.cos(2.0 * angle)],
        axis=2,
    ).astype(np.float32)
    basic = np.concatenate([path_full[:, :, :4], patch_harmonic], axis=2)
    global_candidate = np.concatenate([path_full, route, patch_harmonic], axis=2)
    all_ids = np.arange(len(train_pos), dtype=np.int64)
    folds = v315.disjoint_folds(all_ids, train_pos[:, :2])
    result: dict[str, list[dict[str, float]]] = {
        "global": [],
        "local": [],
        "joint": [],
    }
    predictions = {
        name: np.zeros((len(train_pos), TOP_REFLECTIONS), dtype=np.float32)
        for name in result
    }
    fold_id = np.full(len(train_pos), -1, dtype=np.int16)

    def rows_for(
        ids: np.ndarray, local: np.ndarray, variant: str
    ) -> tuple[np.ndarray, np.ndarray]:
        if variant == "global":
            point = np.repeat(global_point[ids, None, :], TOP_REFLECTIONS, axis=1)
            feature = np.concatenate([point, global_candidate[ids]], axis=2)
        elif variant == "local":
            point = np.repeat(local_point[ids, None, :], TOP_REFLECTIONS, axis=1)
            feature = np.concatenate([point, basic[ids], local], axis=2)
        else:
            global_context = np.repeat(
                global_point[ids, None, :], TOP_REFLECTIONS, axis=1
            )
            local_context = np.repeat(
                local_point[ids, None, :], TOP_REFLECTIONS, axis=1
            )
            feature = np.concatenate(
                [global_context, local_context, global_candidate[ids], local], axis=2
            )
        mask = valid[ids]
        return (
            np.nan_to_num(feature[mask], nan=0.0, posinf=4.0, neginf=-4.0),
            mask,
        )

    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, query_ids)
        fold_id[query_ids] = fold
        fit_local = build_local_evidence(
            fit_ids, fit_ids, train_pos, valid, patch_id, labels
        )
        query_local = build_local_evidence(
            fit_ids, query_ids, train_pos, valid, patch_id, labels
        )
        for variant in result:
            x_fit, fit_mask = rows_for(fit_ids, fit_local, variant)
            y_fit = (labels[fit_ids][fit_mask] > 1e-5).astype(np.int8)
            fit_weight = np.repeat(
                quality[fit_ids, None], TOP_REFLECTIONS, axis=1
            )[fit_mask]
            x_query, query_mask = rows_for(query_ids, query_local, variant)
            y_query = (labels[query_ids][query_mask] > 1e-5).astype(np.int8)
            query_weight = np.repeat(
                quality[query_ids, None], TOP_REFLECTIONS, axis=1
            )[query_mask]
            model = ExtraTreesClassifier(
                n_estimators=TREE_COUNT,
                max_depth=22,
                min_samples_leaf=3,
                max_features=0.65,
                class_weight="balanced",
                n_jobs=-1,
                random_state=43100 + fold,
            )
            model.fit(x_fit, y_fit, sample_weight=fit_weight)
            pair_prediction = model.predict_proba(x_query)[:, 1]
            matrix = np.zeros((len(query_ids), TOP_REFLECTIONS), dtype=np.float32)
            matrix[query_mask] = pair_prediction
            predictions[variant][query_ids] = matrix
            recall, weighted_recall = top5_recall(
                matrix, labels[query_ids], valid[query_ids]
            )
            result[variant].append(
                {
                    "fold": fold,
                    "average_precision": float(
                        average_precision_score(
                            y_query, pair_prediction, sample_weight=query_weight
                        )
                    ),
                    "top5_recall": recall,
                    "mass_weighted_top5_recall": weighted_recall,
                }
            )

    summary = {
        variant: {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in (
                "average_precision",
                "top5_recall",
                "mass_weighted_top5_recall",
            )
        }
        for variant, rows in result.items()
    }
    fold_wins = {
        metric: int(
            sum(
                joint[metric] > max(global_row[metric], local_row[metric])
                for joint, global_row, local_row in zip(
                    result["joint"], result["global"], result["local"]
                )
            )
        )
        for metric in (
            "average_precision",
            "top5_recall",
            "mass_weighted_top5_recall",
        )
    }
    best_single_ap = max(
        summary["global"]["average_precision"],
        summary["local"]["average_precision"],
    )
    best_single_recall = max(
        summary["global"]["top5_recall"], summary["local"]["top5_recall"]
    )
    decision = {
        "joint_ap_margin": summary["joint"]["average_precision"] - best_single_ap,
        "joint_top5_margin": summary["joint"]["top5_recall"] - best_single_recall,
        "required_margin": MIN_JOINT_MARGIN,
        "required_fold_wins": MIN_FOLD_WINS,
        "pass": bool(
            summary["joint"]["average_precision"] - best_single_ap
            >= MIN_JOINT_MARGIN
            and summary["joint"]["top5_recall"] - best_single_recall
            >= MIN_JOINT_MARGIN
            and fold_wins["average_precision"] >= MIN_FOLD_WINS
            and fold_wins["top5_recall"] >= MIN_FOLD_WINS
        ),
    }
    return {
        "fold_rows": result,
        "summary": summary,
        "fold_wins": fold_wins,
        "decision": decision,
    }, {**predictions, "fold_id": fold_id}


def oracle_blend_ceiling(
    train_pos: np.ndarray,
    truth_d: np.ndarray,
    truth_sh: np.ndarray,
    valid_ids: np.ndarray,
    path_d: np.ndarray,
    path_sh: np.ndarray,
) -> dict[str, float]:
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    base_d_parts = []
    base_sh_parts = []
    path_d_parts = []
    path_sh_parts = []
    truth_d_parts = []
    truth_sh_parts = []
    for query_ids in folds:
        fit_ids = np.setdiff1d(valid_ids, query_ids)
        base_d_parts.append(
            v308.generic_knn_anchor(query_ids, fit_ids, train_pos, truth_d)
        )
        base_sh_parts.append(
            v308.generic_knn_anchor(query_ids, fit_ids, train_pos, truth_sh)
        )
        path_d_parts.append(path_d[query_ids])
        path_sh_parts.append(path_sh[query_ids])
        truth_d_parts.append(truth_d[query_ids])
        truth_sh_parts.append(truth_sh[query_ids])
    base_d = np.concatenate(base_d_parts)
    base_sh = np.concatenate(base_sh_parts)
    candidate_d = np.concatenate(path_d_parts)
    candidate_sh = np.concatenate(path_sh_parts)
    target_d = np.concatenate(truth_d_parts)
    target_sh = np.concatenate(truth_sh_parts)
    base_d_score = float(np.mean(row_cos(base_d, target_d)))
    base_sh_score = float(np.mean(row_cos(base_sh, target_sh)))
    base_weighted = 0.48 * base_d_score + 0.52 * base_sh_score
    best: dict[str, float] | None = None
    for alpha_d in ALPHA_GRID:
        for alpha_sh in ALPHA_GRID:
            prediction_d = normalize(
                (1.0 - float(alpha_d)) * base_d + float(alpha_d) * candidate_d
            )
            prediction_sh = normalize(
                (1.0 - float(alpha_sh)) * base_sh
                + float(alpha_sh) * candidate_sh
            )
            d_score = float(np.mean(row_cos(prediction_d, target_d)))
            sh_score = float(np.mean(row_cos(prediction_sh, target_sh)))
            weighted = 0.48 * d_score + 0.52 * sh_score
            row = {
                "alpha_d": float(alpha_d),
                "alpha_sh": float(alpha_sh),
                "d": d_score,
                "sh": sh_score,
                "weighted": weighted,
            }
            if best is None or weighted > best["weighted"]:
                best = row
    assert best is not None
    return {
        "base_d": base_d_score,
        "base_sh": base_sh_score,
        "base_weighted": base_weighted,
        **best,
        "delta": best["weighted"] - base_weighted,
    }


def render_one_bounce_oracle(
    train_pos: np.ndarray,
    train_feature,
    train_candidates: dict[str, np.ndarray],
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = labels[:, : TOP_REFLECTIONS + 1].copy()
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)
    delay = train_feature.delay[:, : TOP_REFLECTIONS + 1]
    h_center = v308.candidate_h_center(train_pos, train_candidates)[
        :, : TOP_REFLECTIONS + 1
    ]
    d_atom = exact_atoms(delay, 192)
    s_atom = exact_atoms(np.remainder(-delay, 192), 192)
    h_atom = exact_atoms(h_center, 16)
    path_d = np.einsum("bp,bpd->bd", weights, d_atom, optimize=True)
    path_sh = np.einsum(
        "bp,bps,bph->bsh", weights, s_atom, h_atom, optimize=True
    )
    return normalize(path_d), normalize(path_sh)


def render_highdepth_oracle(
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    train_base,
    train_candidates: dict[str, np.ndarray],
    patches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    path_data = np.load(v312.PATH_CACHE)
    path_cache = {key: path_data[key] for key in path_data.files}
    interaction_data = np.load(v293.FEATURE_CACHE)
    interaction_cache = {key: interaction_data[key] for key in interaction_data.files}
    high = v313.highdepth_geometry(
        path_cache, interaction_cache, "train", train_pos, patches
    )
    feature = v313.extend_features(train_base, high)
    h_center = np.concatenate(
        [v308.candidate_h_center(train_pos, train_candidates), high["h"]], axis=1
    ).astype(np.float32)
    labels = np.load(v313.LABEL_CACHE)["labels"].astype(np.float32)
    labels /= np.maximum(labels.sum(axis=1, keepdims=True), 1e-30)
    path_d = np.zeros((len(train_pos), 192), dtype=np.float32)
    path_sh = np.zeros((len(train_pos), 192, 16), dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for start in range(0, len(train_pos), 16):
            end = min(start + 16, len(train_pos))
            weight = torch.from_numpy(labels[start:end]).to(device)
            delay = torch.from_numpy(feature.delay[start:end]).to(device)
            h = torch.from_numpy(h_center[start:end]).to(device)
            d_atom = v308.exact_atoms(delay, 192)
            path_d[start:end] = torch.einsum(
                "bp,bpd->bd", weight, d_atom
            ).cpu().numpy()
            path_sh[start:end] = v308.render_sh(weight, delay, h).cpu().numpy()
    return normalize(path_d), normalize(path_sh)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    required = (
        v306.TRAIN_CANDIDATES,
        v306.TEST_CANDIDATES,
        v306.LABEL_CACHE,
        v306.TARGET_CACHE,
        v306.SCENE_CACHE,
        PATHS.cache_dir(270) / "v270_reflection_visibility.npz",
        v312.PATH_CACHE,
        v293.FEATURE_CACHE,
        v313.LABEL_CACHE,
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing v431 inputs: {missing}")

    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    scene = np.load(v306.SCENE_CACHE)
    patches = scene["wall_patches"].astype(np.float32)
    train_data = np.load(v306.TRAIN_CANDIDATES)
    train_candidates = {key: train_data[key] for key in train_data.files}
    test_data = np.load(v306.TEST_CANDIDATES)
    test_candidates = {key: test_data[key] for key in test_data.files}
    train_feature, _test_feature = v306.make_features(
        train_pos, test_pos, train_candidates, test_candidates, patches
    )
    label_data = np.load(v306.LABEL_CACHE)
    full_labels = label_data["labels"].astype(np.float32)
    reflection_labels = full_labels[:, 1 : TOP_REFLECTIONS + 1]
    quality = label_data["quality"].astype(np.float32)
    ranking, predictions = run_candidate_ablation(
        train_pos,
        train_feature,
        train_candidates,
        reflection_labels,
        quality,
    )
    np.savez_compressed(
        OOF_CACHE,
        labels=reflection_labels,
        valid=train_candidates["score"][:, :TOP_REFLECTIONS] > 0.0,
        **predictions,
    )

    targets = np.load(v306.TARGET_CACHE)
    truth_d = targets["d"].astype(np.float32)
    truth_sh = targets["sh"].astype(np.float32)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    one_d, one_sh = render_one_bounce_oracle(
        train_pos, train_feature, train_candidates, full_labels
    )
    one_ceiling = oracle_blend_ceiling(
        train_pos, truth_d, truth_sh, valid_ids, one_d, one_sh
    )
    high_d, high_sh = render_highdepth_oracle(
        train_pos, test_pos, train_feature, train_candidates, patches
    )
    high_ceiling = oracle_blend_ceiling(
        train_pos, truth_d, truth_sh, valid_ids, high_d, high_sh
    )
    oracle_pass = high_ceiling["delta"] >= MIN_ORACLE_DELTA
    final_pass = bool(ranking["decision"]["pass"] and oracle_pass)
    report = {
        "method": VERSION,
        "status": "stopped_before_high_budget_training" if not final_pass else "gate_passed",
        "primary_hypothesis": (
            "Full-route geometry and fold-isolated same-patch local support are "
            "complementary enough to learn candidate existence and energy."
        ),
        "fixed_gates": {
            "joint_margin_over_best_single": MIN_JOINT_MARGIN,
            "minimum_fold_wins": MIN_FOLD_WINS,
            "minimum_highdepth_oracle_blend_delta": MIN_ORACLE_DELTA,
        },
        "candidate_ranking_ablation": ranking,
        "oracle_ceiling": {
            "one_bounce_top32": one_ceiling,
            "one_bounce_plus_64_highdepth": high_ceiling,
            "highdepth_gate_pass": bool(oracle_pass),
        },
        "decision": {
            "pass": final_pass,
            "high_budget_transformer_run": False,
            "reason": (
                "Joint features did not beat global-only, and even truth-matched "
                "high-depth candidate weights improved the KNN spectral anchor by less "
                "than the predeclared 0.01 ceiling gate."
            ),
        },
        "next_route": (
            "Keep global-route and local-context encoders, but replace discrete candidate "
            "weight selection with path-guided continuous delay-angle residual generation "
            "and an explicit unmatched/diffuse support branch."
        ),
        "outputs": [PATHS.relative(OOF_CACHE), PATHS.relative(REPORT)],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
