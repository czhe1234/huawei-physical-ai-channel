#!/usr/bin/env python3
"""Learn when official-pointcloud Sionna PDP is useful beyond a KNN anchor."""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v412_path_sequence_identity_student as v412
import analyze_v417_multiresolution_global_d_field as v417
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 418
VERSION = "v418_physics_pdp_trust_gate"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v418_physics_pdp_trust_gate_oof.npz"

PATH_CACHE = PATHS.cache_dir(291) / "v291_highdepth_pointcloud_paths.npz"
COEFFICIENT_CACHE = PATHS.cache_dir(316) / "v316_pointcloud_path_coefficients.npz"
ALPHAS = np.asarray([0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50], dtype=np.float32)
MIN_D_DELTA = 0.005
MIN_FOLD_WINS = 4


def normalize(value: np.ndarray) -> np.ndarray:
    return v417.normalize(value)


def row_cos(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return v417.row_cos(left, right)


def render_physics(prefix: str) -> tuple[np.ndarray, np.ndarray]:
    paths = np.load(PATH_CACHE)
    coefficient = np.load(COEFFICIENT_CACHE)[f"{prefix}_coefficient"]
    delay = paths[f"{prefix}_delay_bin_physical"].astype(np.float64)
    valid = (
        paths[f"{prefix}_valid"].astype(bool)
        & np.isfinite(delay)
        & (np.abs(coefficient) > 0.0)
    )
    safe_delay = np.where(valid, delay, 0.0)
    power = np.abs(coefficient).astype(np.float64) ** 2 * valid
    output = np.zeros((len(delay), 192), dtype=np.float64)
    lower = np.floor(safe_delay).astype(np.int64) % 192
    fraction = safe_delay - np.floor(safe_delay)
    rows = np.arange(len(delay), dtype=np.int64)[:, None]
    np.add.at(output, (rows, lower), power * (1.0 - fraction))
    np.add.at(output, (rows, (lower + 1) % 192), power * fraction)
    total = output.sum(axis=1, keepdims=True)
    available = total[:, 0] > 0.0
    output[available] /= total[available]
    return output.astype(np.float32), available


def entropy(value: np.ndarray) -> np.ndarray:
    probability = normalize(value)
    return (
        -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1)
        / np.log(probability.shape[1])
    ).astype(np.float32)


def circular_moments(value: np.ndarray) -> np.ndarray:
    probability = normalize(value)
    angle = 2.0 * np.pi * np.arange(192, dtype=np.float64) / 192.0
    first = np.sum(probability * np.exp(1j * angle)[None], axis=1)
    second = np.sum(probability * np.exp(2j * angle)[None], axis=1)
    peak = np.argmax(probability, axis=1)
    return np.column_stack(
        [
            first.real,
            first.imag,
            np.abs(first),
            second.real,
            second.imag,
            np.abs(second),
            np.sin(2.0 * np.pi * peak / 192.0),
            np.cos(2.0 * np.pi * peak / 192.0),
            probability.max(axis=1),
            entropy(probability),
        ]
    ).astype(np.float32)


def path_statistics(prefix: str) -> np.ndarray:
    paths = np.load(PATH_CACHE)
    coefficient = np.load(COEFFICIENT_CACHE)[f"{prefix}_coefficient"]
    valid = paths[f"{prefix}_valid"].astype(bool) & (np.abs(coefficient) > 0.0)
    power = np.abs(coefficient).astype(np.float64) ** 2 * valid
    weight = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-300)
    depth = paths[f"{prefix}_interaction_count"].astype(np.float64)
    length = np.nan_to_num(paths[f"{prefix}_path_length"].astype(np.float64))
    score = np.nan_to_num(paths[f"{prefix}_score"].astype(np.float64))
    feature = [
        valid.sum(axis=1),
        np.sum(weight * depth, axis=1),
        np.sum(weight * length, axis=1) / 300.0,
        np.sum(weight * score, axis=1),
        np.max(power, axis=1) / np.maximum(power.sum(axis=1), 1e-300),
        -np.sum(weight * np.log(np.maximum(weight, 1e-30)), axis=1) / np.log(128.0),
    ]
    for boundary in (0, 1, 2, 4, 8, 12):
        if boundary == 0:
            mask = depth == 0
        elif boundary == 12:
            mask = depth >= boundary
        else:
            mask = (depth >= boundary) & (depth < boundary + 2)
        feature.append(np.sum(weight * mask, axis=1))
    return np.column_stack(feature).astype(np.float32)


def observable_features(
    node: np.ndarray,
    anchor: np.ndarray,
    physics: np.ndarray,
    path_feature: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    anchor_moment = circular_moments(anchor)
    physics_moment = circular_moments(physics)
    relation = np.column_stack(
        [
            row_cos(anchor, physics),
            np.sum(np.abs(anchor - physics), axis=1),
            np.mean(np.abs(np.cumsum(np.roll(anchor - physics, -96, axis=1), axis=1)), axis=1),
            entropy(anchor),
            entropy(physics),
            available.astype(np.float32),
        ]
    ).astype(np.float32)
    return np.nan_to_num(
        np.concatenate(
            [node, anchor_moment, physics_moment, physics_moment - anchor_moment, relation, path_feature],
            axis=1,
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def mixtures(anchor: np.ndarray, physics: np.ndarray) -> np.ndarray:
    return np.stack(
        [normalize((1.0 - alpha) * anchor + alpha * physics) for alpha in ALPHAS],
        axis=1,
    )


def oracle_alpha(anchor: np.ndarray, physics: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidates = mixtures(anchor, physics)
    score = np.stack([row_cos(candidates[:, index], truth) for index in range(len(ALPHAS))], axis=1)
    best = np.argmax(score, axis=1)
    return ALPHAS[best], score[np.arange(len(score)), best] - row_cos(anchor, truth)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    targets = np.load(v412.TARGET_CACHE)
    truth_all = normalize(targets["d"])
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    xy = train_pos[:, :2]
    node, _ = v257.build_node_features(xy, test_pos[:, :2])
    physics, available = render_physics("train")
    path_feature = path_statistics("train")
    folds = v315.disjoint_folds(valid_ids, xy)
    fold_rows: list[dict[str, object]] = []
    parts: list[dict[str, np.ndarray]] = []
    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, query_ids)
        fit_anchor = v417.knn_anchor(fit_ids, fit_ids, xy, truth_all)
        outer_anchor = v417.knn_anchor(query_ids, fit_ids, xy, truth_all)
        fit_feature = observable_features(
            node[fit_ids], fit_anchor, physics[fit_ids], path_feature[fit_ids], available[fit_ids]
        )
        outer_feature = observable_features(
            node[query_ids], outer_anchor, physics[query_ids], path_feature[query_ids], available[query_ids]
        )
        fit_label, fit_oracle = oracle_alpha(
            fit_anchor, physics[fit_ids], truth_all[fit_ids]
        )
        model = ExtraTreesRegressor(
            n_estimators=700,
            max_depth=18,
            min_samples_leaf=8,
            max_features=0.65,
            random_state=41800 + fold,
            n_jobs=8,
        )
        model.fit(fit_feature, fit_label, sample_weight=0.25 + np.maximum(fit_oracle, 0.0))
        alpha = np.clip(model.predict(outer_feature), 0.0, 0.5).astype(np.float32)
        alpha[~available[query_ids]] = 0.0
        prediction = normalize(
            (1.0 - alpha[:, None]) * outer_anchor + alpha[:, None] * physics[query_ids]
        )
        truth = truth_all[query_ids]
        base_score = row_cos(outer_anchor, truth)
        score = row_cos(prediction, truth)
        _, outer_oracle = oracle_alpha(outer_anchor, physics[query_ids], truth)
        row = {
            "fold": fold,
            "fit_count": int(len(fit_ids)),
            "query_count": int(len(query_ids)),
            "cosine_delta": float(np.mean(score - base_score)),
            "positive_fraction": float(np.mean(score > base_score)),
            "oracle_ceiling": float(np.mean(outer_oracle)),
            "predicted_alpha": summarize(alpha),
        }
        fold_rows.append(row)
        parts.append(
            {
                "ids": query_ids,
                "baseline": outer_anchor,
                "truth": truth,
                "physics": physics[query_ids],
                "prediction": prediction,
                "alpha": alpha,
                "point_delta": score - base_score,
            }
        )
        print(f"v418 fold={fold} delta={row['cosine_delta']:+.6f}", flush=True)
    delta = np.asarray([row["cosine_delta"] for row in fold_rows], dtype=np.float64)
    passed = bool(np.mean(delta) >= MIN_D_DELTA and np.sum(delta > 0.0) >= MIN_FOLD_WINS)
    ids = np.concatenate([part["ids"] for part in parts])
    if len(ids) != 1984 or len(np.unique(ids)) != 1984:
        raise AssertionError("v418 OOF IDs are not a unique 1984-point cover")
    np.savez_compressed(
        OOF_CACHE,
        **{key: np.concatenate([part[key] for part in parts]) for key in parts[0]},
        fold=np.concatenate([np.full(len(part["ids"]), index, dtype=np.int16) for index, part in enumerate(parts)]),
    )
    report = {
        "method": VERSION,
        "status": "official_oof_passed" if passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Official-pointcloud Sionna path coefficients propose useful PDP only for a subset of points; "
            "an official-train-H gate can predict its trust weight."
        ),
        "compliance": {
            "uses": [
                "official train H normalized D",
                "official train/test coordinates and pointcloud mesh",
                "Sionna default-concrete path coefficients as AI input features",
            ],
            "does_not_use": [
                "synthetic test target or v212 teacher",
                "restored Etoile scene/material/object labels",
                "raw RT H as the final prediction",
                "online labels",
            ],
        },
        "fold_rows": fold_rows,
        "aggregate": {
            "cosine_delta": summarize(delta),
            "positive_folds": int(np.sum(delta > 0.0)),
        },
        "selection": {
            "minimum_cosine_delta": MIN_D_DELTA,
            "minimum_positive_folds": MIN_FOLD_WINS,
            "passed": passed,
            "next_action": "train final gate" if passed else "do not deploy this gate",
        },
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    for path in (PATH_CACHE, COEFFICIENT_CACHE, v412.TARGET_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    run()


if __name__ == "__main__":
    main()
