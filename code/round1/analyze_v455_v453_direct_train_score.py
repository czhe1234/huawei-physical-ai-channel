#!/usr/bin/env python3
"""Compute the direct official-train fitting score of the v453 adapter."""

from __future__ import annotations

import json
import time

import numpy as np

import analyze_v451_official_path_representation_probe as v451
import analyze_v452_v445_representation_over_strong_anchor as v452
import analyze_v454_v453_official_fullh_local_score as v454
from project_paths import PATHS


VERSION_NUMBER = 455
VERSION = "v455_v453_direct_train_score"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
DETAILS = OUT_DIR / "v455_direct_train_scores.npz"

TARGET_SOURCE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOMETRY_SOURCE = PATHS.shared_features / "height_geometry_cache.npz"
EMBEDDING_SOURCE = PATHS.cache_dir(451) / "v445_path_embedding.npy"
V98_BASE_AD = PATHS.shared_features / "train_v98like_loo_base_ad.npy"
V140_BASE_AD = PATHS.cache_dir(156) / "train_v140eq_crossfit_base_ad.npy"

RIDGE = 10.0
STRENGTH = 0.15
BATCH = 8


def fit_direct_delta() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    target = np.load(TARGET_SOURCE)
    truth_sh = target["sh"].astype(np.float32)
    truth_d = target["d"].astype(np.float32)
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    blocked = np.load(GEOMETRY_SOURCE)["train_blocked"].astype(np.float32)
    feature = np.load(EMBEDDING_SOURCE).astype(np.float32)[:2000]

    anchor = v452.knn_anchor(
        valid_ids,
        valid_ids,
        positions,
        blocked,
        truth_sh,
        exclude_self=True,
    )
    residual = v452.residual_target(truth_sh[valid_ids], anchor)
    local_ids = np.arange(len(valid_ids), dtype=np.int64)
    latent, pca = v451.prepare_latent_target(residual, local_ids, 453)

    train_feature = feature[valid_ids]
    mean = np.mean(train_feature, axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(
        np.std(train_feature, axis=0, dtype=np.float64).astype(np.float32), 1e-4
    )
    standardized = ((train_feature - mean) / std).astype(np.float32)
    coefficient, intercept = v451.ridge_fit(standardized, latent, RIDGE)
    prediction = standardized @ coefficient + intercept
    prediction = pca.inverse_transform(prediction).astype(np.float32)
    prediction = np.clip(prediction, -6.0, 6.0)

    delta = np.zeros((2000, 192, 16), dtype=np.float32)
    delta[valid_ids] = prediction.reshape(-1, 192, 16)
    model_support = v451.apply_delta(anchor, prediction, STRENGTH)
    support_base = v452.score_joint(anchor, truth_sh[valid_ids], truth_d[valid_ids])
    support_model = v452.score_joint(
        model_support, truth_sh[valid_ids], truth_d[valid_ids]
    )
    return delta, valid_ids, {
        "base": support_base,
        "model": support_model,
        "delta": v452.delta_metric(support_model, support_base),
    }


def evaluate_carrier(
    name: str, path, delta: np.ndarray, valid_ids: np.ndarray
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    base_ad = np.load(path, mmap_mode="r")
    truth_h = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    base_values = v454.empty_point_metrics(2000)
    model_values = v454.empty_point_metrics(2000)
    action = np.zeros(2000, dtype=np.float32)

    for start in range(0, 2000, BATCH):
        end = min(start + BATCH, 2000)
        ids = np.arange(start, end, dtype=np.int64)
        base_h = v454.restore_raw_h(
            np.asarray(base_ad[start:end], dtype=np.complex64), positions[start:end]
        )
        model_h, action[ids] = v454.apply_delta(base_h, delta[start:end])
        truth = np.asarray(truth_h[start:end], dtype=np.complex64)
        v454.add_scores(base_values, ids, base_h, truth)
        v454.add_scores(model_values, ids, model_h, truth)
        print(f"v455 {name} direct train score {end}/2000", flush=True)

    all_ids = np.arange(2000, dtype=np.int64)
    base_valid = v454.aggregate(base_values, valid_ids)
    model_valid = v454.aggregate(model_values, valid_ids)
    base_all = v454.aggregate(base_values, all_ids)
    model_all = v454.aggregate(model_values, all_ids)
    result = {
        "carrier": name,
        "source": PATHS.relative(path),
        "valid_1984": {
            "base": base_valid,
            "model": model_valid,
            "delta": v454.delta_metric(model_valid, base_valid),
        },
        "all_2000": {
            "base": base_all,
            "model": model_all,
            "delta": v454.delta_metric(model_all, base_all),
        },
        "action_nmse": v454.summarize(action[valid_ids]),
    }
    detail = {
        f"{name}_base_pas": base_values["pas"],
        f"{name}_base_pdp": base_values["pdp"],
        f"{name}_base_error": base_values["error"],
        f"{name}_model_pas": model_values["pas"],
        f"{name}_model_pdp": model_values["pdp"],
        f"{name}_model_error": model_values["error"],
        f"{name}_energy": model_values["energy"],
        f"{name}_action_nmse": action,
    }
    return result, detail


def main() -> None:
    required = (
        TARGET_SOURCE,
        GEOMETRY_SOURCE,
        EMBEDDING_SOURCE,
        V98_BASE_AD,
        V140_BASE_AD,
        PATHS.round1 / "Round1_Train_Channel.npy",
        PATHS.round1 / "Round1_Train_Pos.npy",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    delta, valid_ids, support = fit_direct_delta()
    carriers = {}
    details: dict[str, np.ndarray] = {
        "valid_ids": valid_ids.astype(np.int32),
        "direct_train_delta": delta.astype(np.float16),
    }
    for name, path in (
        ("v98like_loo", V98_BASE_AD),
        ("v140eq_crossfit", V140_BASE_AD),
    ):
        result, carrier_details = evaluate_carrier(name, path, delta, valid_ids)
        carriers[name] = result
        details.update(carrier_details)
    np.savez_compressed(DETAILS, **details)

    report = {
        "method": VERSION,
        "status": "analysis_only_direct_train_fit_score",
        "primary_question": (
            "What score does the v453 adapter obtain when trained on all official "
            "train rows and evaluated directly on those same rows?"
        ),
        "metric": {
            "formula": "0.4*PAS + 0.4*PDP + 0.2/(1+global_NMSE)",
            "truth": "official Round1_Train_Channel.npy only",
            "evaluation": "direct train fit; no OOF isolation for the residual model",
        },
        "adapter": {
            "ridge": RIDGE,
            "strength": STRENGTH,
            "valid_train_count": int(len(valid_ids)),
            "support_direct_fit": support,
        },
        "carriers": carriers,
        "interpretation_boundary": (
            "This is a direct training-fit statistic requested by the user. It is "
            "not a generalization estimate and still requires a full-H carrier because "
            "v453 predicts only a joint support correction."
        ),
        "seconds": float(time.time() - started),
        "outputs": {
            "details": PATHS.relative(DETAILS),
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
