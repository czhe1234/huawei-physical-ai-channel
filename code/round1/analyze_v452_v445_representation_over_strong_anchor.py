#!/usr/bin/env python3
"""Test v445 path representations on residuals over the strong official KNN anchor."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import analyze_v401_dualdomain_pas_preserving_confocal as v401
import analyze_v451_official_path_representation_probe as v451
import generate_v218_official_multispectrum_physics_ai as v218
from project_paths import PATHS


VERSION_NUMBER = 452
VERSION = "v452_v445_representation_over_strong_anchor"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v452_strong_anchor_path_representation_oof.npz"

TARGET_SOURCE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOMETRY_SOURCE = PATHS.shared_features / "height_geometry_cache.npz"
RIDGE_GRID = v451.RIDGE_GRID
STRENGTH_GRID = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30)
FLOOR = v451.FLOOR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=45_201)
    return parser.parse_args()


def quiet_weighted_average(
    target: np.ndarray, index: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    shape = (len(index), index.shape[1]) + (1,) * (target.ndim - 1)
    value = np.sum(target[index] * weight.reshape(shape), axis=1, dtype=np.float64)
    value = np.maximum(value.astype(np.float32), 0.0)
    denominator = np.sum(
        value, axis=tuple(range(1, value.ndim)), keepdims=True, dtype=np.float64
    )
    return (value / np.maximum(denominator, 1e-30)).astype(np.float32)


def knn_anchor(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    positions: np.ndarray,
    blocked: np.ndarray,
    target: np.ndarray,
    *,
    exclude_self: bool,
) -> np.ndarray:
    query_identity = query_ids if exclude_self else None
    bank_identity = bank_ids if exclude_self else None
    index, weight = v218.neighbor_index_weights(
        positions[query_ids],
        blocked[query_ids],
        positions[bank_ids],
        blocked[bank_ids],
        query_ids=query_identity,
        bank_ids=bank_identity,
        k=v218.ANCHOR_K,
    )
    return quiet_weighted_average(target[bank_ids], index, weight)


def score_joint(
    joint: np.ndarray, truth_sh: np.ndarray, truth_d: np.ndarray
) -> dict[str, float]:
    reverse = np.concatenate([[0], np.arange(191, 0, -1)])
    pdp = np.sum(joint[:, reverse], axis=2)
    sh = float(np.mean(v451.v450.row_cosine(joint, truth_sh)))
    d = float(np.mean(v451.v450.row_cosine(pdp, truth_d)))
    return {
        "score_60_joint_40_pdp": 0.60 * sh + 0.40 * d,
        "weighted_52_sh_48_d": 0.52 * sh + 0.48 * d,
        "joint_sh_cosine": sh,
        "pdp_cosine": d,
    }


def delta_metric(model: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {key: float(model[key] - base[key]) for key in model}


def residual_target(target: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    residual = np.log(np.maximum(target.reshape(len(target), -1), FLOOR)) - np.log(
        np.maximum(anchor.reshape(len(anchor), -1), FLOOR)
    )
    return np.clip(residual, -6.0, 6.0).astype(np.float32)


def build_fold_targets(
    folds: list[np.ndarray],
    valid_ids: np.ndarray,
    positions: np.ndarray,
    blocked: np.ndarray,
    truth_sh: np.ndarray,
    seed: int,
) -> list[dict[str, object]]:
    output = []
    for outer, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, query_ids)
        validation_ids = np.intersect1d(fit_ids, folds[(outer + 1) % len(folds)])
        inner_train_ids = np.setdiff1d(fit_ids, validation_ids)

        inner_train_anchor = knn_anchor(
            inner_train_ids,
            inner_train_ids,
            positions,
            blocked,
            truth_sh,
            exclude_self=True,
        )
        validation_anchor = knn_anchor(
            validation_ids,
            inner_train_ids,
            positions,
            blocked,
            truth_sh,
            exclude_self=False,
        )
        fit_anchor = knn_anchor(
            fit_ids,
            fit_ids,
            positions,
            blocked,
            truth_sh,
            exclude_self=True,
        )
        query_anchor = knn_anchor(
            query_ids,
            fit_ids,
            positions,
            blocked,
            truth_sh,
            exclude_self=False,
        )
        inner_residual = residual_target(
            truth_sh[inner_train_ids], inner_train_anchor
        )
        full_residual = residual_target(truth_sh[fit_ids], fit_anchor)
        inner_latent, inner_pca = v451.prepare_latent_target(
            inner_residual,
            np.arange(len(inner_train_ids), dtype=np.int64),
            seed + 1000 * outer,
        )
        full_latent, full_pca = v451.prepare_latent_target(
            full_residual,
            np.arange(len(fit_ids), dtype=np.int64),
            seed + 20_000 + 1000 * outer,
        )
        output.append(
            {
                "query_ids": query_ids,
                "fit_ids": fit_ids,
                "validation_ids": validation_ids,
                "inner_train_ids": inner_train_ids,
                "validation_anchor": validation_anchor,
                "query_anchor": query_anchor,
                "inner_latent": inner_latent,
                "inner_pca": inner_pca,
                "full_latent": full_latent,
                "full_pca": full_pca,
            }
        )
        print(
            f"v452 prepared fold={outer} fit={len(fit_ids)} query={len(query_ids)}",
            flush=True,
        )
    return output


def fit_predict_low_rank(
    feature: np.ndarray,
    train_ids: np.ndarray,
    query_ids: np.ndarray,
    target_latent: np.ndarray,
    pca,
    ridge: float,
) -> np.ndarray:
    fit_x, query_x = v451.standardize_fit(feature, train_ids, query_ids)
    coefficient, intercept = v451.ridge_fit(fit_x, target_latent, ridge)
    latent = query_x @ coefficient + intercept
    return pca.inverse_transform(latent).astype(np.float32)


def probe_method(
    name: str,
    feature: np.ndarray,
    fold_targets: list[dict[str, object]],
    valid_ids: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    oof_base = np.zeros_like(truth_sh)
    oof_model = np.zeros_like(truth_sh)
    fold_rows = []
    for outer, fold_target in enumerate(fold_targets):
        query_ids = fold_target["query_ids"]
        fit_ids = fold_target["fit_ids"]
        validation_ids = fold_target["validation_ids"]
        inner_train_ids = fold_target["inner_train_ids"]
        validation_anchor = fold_target["validation_anchor"]
        query_anchor = fold_target["query_anchor"]
        best = None
        for ridge in RIDGE_GRID:
            validation_delta = fit_predict_low_rank(
                feature,
                inner_train_ids,
                validation_ids,
                fold_target["inner_latent"],
                fold_target["inner_pca"],
                ridge,
            )
            for strength in STRENGTH_GRID:
                prediction = v451.apply_delta(
                    validation_anchor, validation_delta, strength
                )
                metric = score_joint(
                    prediction, truth_sh[validation_ids], truth_d[validation_ids]
                )
                candidate = (
                    metric["weighted_52_sh_48_d"],
                    -ridge,
                    -strength,
                    ridge,
                    strength,
                )
                if best is None or candidate > best[0]:
                    best = (candidate, metric)
        assert best is not None
        ridge = float(best[0][3])
        strength = float(best[0][4])
        query_delta = fit_predict_low_rank(
            feature,
            fit_ids,
            query_ids,
            fold_target["full_latent"],
            fold_target["full_pca"],
            ridge,
        )
        prediction = v451.apply_delta(query_anchor, query_delta, strength)
        oof_base[query_ids] = query_anchor
        oof_model[query_ids] = prediction
        base_metric = score_joint(
            query_anchor, truth_sh[query_ids], truth_d[query_ids]
        )
        model_metric = score_joint(
            prediction, truth_sh[query_ids], truth_d[query_ids]
        )
        fold_rows.append(
            {
                "fold": outer,
                "fit_count": int(len(fit_ids)),
                "query_count": int(len(query_ids)),
                "selected_ridge": ridge,
                "selected_strength": strength,
                "inner_validation": best[1],
                "base": base_metric,
                "model": model_metric,
                "delta": delta_metric(model_metric, base_metric),
            }
        )
        print(
            f"v452 {name} fold={outer} ridge={ridge:.0f} strength={strength:.3f} "
            f"delta={fold_rows[-1]['delta']['weighted_52_sh_48_d']:+.6f}",
            flush=True,
        )
    base_metric = score_joint(
        oof_base[valid_ids], truth_sh[valid_ids], truth_d[valid_ids]
    )
    model_metric = score_joint(
        oof_model[valid_ids], truth_sh[valid_ids], truth_d[valid_ids]
    )
    return {
        "feature_dim": int(feature.shape[1]),
        "base": base_metric,
        "model": model_metric,
        "delta": delta_metric(model_metric, base_metric),
        "positive_folds": int(
            sum(
                row["delta"]["weighted_52_sh_48_d"] > 0.0
                for row in fold_rows
            )
        ),
        "mean_selected_strength": float(
            np.mean([row["selected_strength"] for row in fold_rows])
        ),
        "folds": fold_rows,
    }, oof_base, oof_model


def main() -> None:
    args = parse_args()
    for path in (TARGET_SOURCE, GEOMETRY_SOURCE):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    input_data = v451.v450.load_or_build_inputs()
    embedding_paths = {
        "raw_geometry": PATHS.cache_dir(451) / "raw_geometry_features.npy",
        "random_path_encoder": PATHS.cache_dir(451) / "random_path_embedding.npy",
        "v445_path_encoder": PATHS.cache_dir(451) / "v445_path_embedding.npy",
    }
    for path in embedding_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    embeddings = {
        name: np.load(path).astype(np.float32)
        for name, path in embedding_paths.items()
    }
    embeddings["raw_plus_v445"] = np.concatenate(
        [embeddings["raw_geometry"], embeddings["v445_path_encoder"]], axis=1
    ).astype(np.float32)
    target = np.load(TARGET_SOURCE)
    truth_sh = target["sh"].astype(np.float32)
    truth_d = target["d"].astype(np.float32)
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    positions = np.asarray(input_data["positions"][:2000], dtype=np.float32)
    geometry = np.load(GEOMETRY_SOURCE)
    blocked = geometry["train_blocked"].astype(np.float32)
    folds = v401.confirmatory_folds(valid_ids, positions[:, :2])
    fold_targets = build_fold_targets(
        folds, valid_ids, positions, blocked, truth_sh, args.seed
    )
    rng = np.random.default_rng(args.seed)
    methods = {}
    predictions = {}
    shared_base = None
    for name, feature in embeddings.items():
        result, base, prediction = probe_method(
            name,
            feature[:2000],
            fold_targets,
            valid_ids,
            truth_sh,
            truth_d,
        )
        methods[name] = result
        predictions[name] = prediction
        if shared_base is None:
            shared_base = base
    shuffled = embeddings["v445_path_encoder"][:2000][rng.permutation(2000)]
    shuffled_result, _, shuffled_prediction = probe_method(
        "shuffled_v445_path_encoder",
        shuffled,
        fold_targets,
        valid_ids,
        truth_sh,
        truth_d,
    )
    methods["shuffled_v445_path_encoder"] = shuffled_result
    predictions["shuffled_v445_path_encoder"] = shuffled_prediction
    winner = max(
        methods, key=lambda name: methods[name]["delta"]["weighted_52_sh_48_d"]
    )
    v445_gain = float(
        methods["v445_path_encoder"]["delta"]["weighted_52_sh_48_d"]
    )
    raw_gain = float(methods["raw_geometry"]["delta"]["weighted_52_sh_48_d"])
    random_gain = float(
        methods["random_path_encoder"]["delta"]["weighted_52_sh_48_d"]
    )
    passed = bool(
        v445_gain >= 5e-4
        and methods["v445_path_encoder"]["positive_folds"] >= 4
        and v445_gain > max(raw_gain, random_gain)
    )
    assert shared_base is not None
    np.savez_compressed(
        OOF_CACHE,
        valid_ids=valid_ids,
        base=shared_base[valid_ids].astype(np.float16),
        **{
            name: prediction[valid_ids].astype(np.float16)
            for name, prediction in predictions.items()
        },
    )
    report = {
        "method": VERSION,
        "status": "strict_official_oof_strong_anchor_probe",
        "primary_hypothesis": (
            "The transferable v445 path representation predicts residual joint support "
            "that the established official-only 72-neighbor geometry anchor misses."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H-derived targets",
                "official train coordinates and point-cloud blocked features",
                "official point-cloud geometry path embeddings",
                "strict nested five-fold residual training over a fit-only KNN anchor",
            ],
            "does_not_use": [
                "official test H",
                "test-coordinate Sionna CFR or path coefficients",
                "v436/v449 teacher outputs",
                "online score selection",
            ],
        },
        "parameters": {
            "anchor_k": v218.ANCHOR_K,
            "ridge_grid": list(RIDGE_GRID),
            "strength_grid": list(STRENGTH_GRID),
            "target_latent_dim": v451.LATENT_DIM,
            "seed": args.seed,
        },
        "methods": methods,
        "winner": winner,
        "decision": {
            "criterion": (
                "v445 representation must add >=0.0005 weighted support, win >=4/5 "
                "folds, and beat raw/random geometry encoders"
            ),
            "passed": passed,
            "next_step": (
                "fit the official adapter on all train H and project a strict carrier"
                if passed
                else "do not generate a candidate from the current representation"
            ),
        },
        "seconds": float(time.time() - started),
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
