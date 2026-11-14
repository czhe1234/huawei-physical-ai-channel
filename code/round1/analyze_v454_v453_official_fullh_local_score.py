#!/usr/bin/env python3
"""Compute the official-train full-H local score for the v453 adapter."""

from __future__ import annotations

import json
import time

import numpy as np

import analyze_v205_sionna_config_calibration as metric
import analyze_v401_dualdomain_pas_preserving_confocal as v401
import analyze_v451_official_path_representation_probe as v451
import analyze_v452_v445_representation_over_strong_anchor as v452
import generate_v187_leaked_pas_lifecycle_projection as leak
from generate_v171_full_complex_ad_tail_denoising import top_mask
from generate_v87_easy_field_projection import BS_POS, beta_subcarriers, inverse_ad
from project_paths import PATHS


VERSION_NUMBER = 454
VERSION = "v454_v453_official_fullh_local_score"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
DETAILS = OUT_DIR / "v454_official_fullh_oof_scores.npz"

TARGET_SOURCE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOMETRY_SOURCE = PATHS.shared_features / "height_geometry_cache.npz"
EMBEDDING_SOURCE = PATHS.cache_dir(451) / "v445_path_embedding.npy"
V98_BASE_AD = PATHS.shared_features / "train_v98like_loo_base_ad.npy"
V140_BASE_AD = PATHS.cache_dir(156) / "train_v140eq_crossfit_base_ad.npy"

RIDGE = 10.0
STRENGTH = 0.15
SEED = 45_201
BATCH = 8
FLOOR = v451.FLOOR
TINY = 1e-30


def summarize(value: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def fixed_oof_delta() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    target = np.load(TARGET_SOURCE)
    truth_sh = target["sh"].astype(np.float32)
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    geometry = np.load(GEOMETRY_SOURCE)
    blocked = geometry["train_blocked"].astype(np.float32)
    feature = np.load(EMBEDDING_SOURCE).astype(np.float32)[:2000]

    folds = v401.confirmatory_folds(valid_ids, positions[:, :2])
    fold_targets = v452.build_fold_targets(
        folds, valid_ids, positions, blocked, truth_sh, SEED
    )
    delta = np.zeros((2000, 192, 16), dtype=np.float32)
    fold_index = np.full(2000, -1, dtype=np.int16)
    fold_rows: list[dict[str, object]] = []
    for outer, fold_target in enumerate(fold_targets):
        query_ids = np.asarray(fold_target["query_ids"], dtype=np.int64)
        fit_ids = np.asarray(fold_target["fit_ids"], dtype=np.int64)
        prediction = v452.fit_predict_low_rank(
            feature,
            fit_ids,
            query_ids,
            fold_target["full_latent"],
            fold_target["full_pca"],
            RIDGE,
        )
        prediction = np.clip(prediction, -6.0, 6.0)
        delta[query_ids] = prediction.reshape(-1, 192, 16)
        fold_index[query_ids] = outer

        anchor = np.asarray(fold_target["query_anchor"], dtype=np.float32)
        model = v451.apply_delta(anchor, prediction, STRENGTH)
        truth_d = target["d"].astype(np.float32)
        base_metric = v452.score_joint(anchor, truth_sh[query_ids], truth_d[query_ids])
        model_metric = v452.score_joint(model, truth_sh[query_ids], truth_d[query_ids])
        fold_rows.append(
            {
                "fold": outer,
                "query_count": int(len(query_ids)),
                "support_base": base_metric,
                "support_model": model_metric,
                "support_delta": v452.delta_metric(model_metric, base_metric),
            }
        )
        print(f"v454 fixed residual fold {outer + 1}/{len(folds)}", flush=True)

    if np.any(fold_index[valid_ids] < 0):
        raise ValueError("Not all valid official train rows received OOF predictions")
    return delta, valid_ids, fold_index, {"folds": fold_rows}


def restore_raw_h(ad: np.ndarray, positions: np.ndarray) -> np.ndarray:
    derotated = inverse_ad(ad)
    distance = np.linalg.norm(positions - BS_POS[None, :], axis=1)
    beta_s = beta_subcarriers(140.3, 0.10)
    derotated *= np.exp(
        -1j * distance[:, None, None, None] * beta_s[None, None, None, :]
    ).astype(np.complex64)
    return derotated.astype(np.complex64)


def apply_delta(h0: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spec0 = leak.leaked_forward(h0)
    power0 = leak.leaked_cell_power(spec0)
    total0 = np.maximum(
        np.sum(power0, axis=(1, 2, 3, 4, 5), dtype=np.float64), TINY
    ).astype(np.float32)
    support0 = power0 / total0[:, None, None, None, None, None]
    current_sh = np.sum(support0, axis=(1, 4, 5), dtype=np.float64).astype(
        np.float32
    )
    ratio = np.exp(np.clip(STRENGTH * delta, -1.8, 1.8)).astype(np.float32)
    desired_sh = current_sh * ratio
    desired_sh /= np.maximum(
        np.sum(desired_sh, axis=(1, 2), keepdims=True, dtype=np.float64), TINY
    )
    cell_ratio = desired_sh / np.maximum(current_sh, FLOOR)
    gain = np.sqrt(cell_ratio)[:, None, :, :, None, None].astype(np.float32)
    gain = np.broadcast_to(gain, support0.shape).copy()
    gain = np.clip(gain, 0.32, 3.20)

    flat = support0.reshape(len(h0), -1)
    strong128 = top_mask(flat, 128).reshape(support0.shape)
    strong1024 = top_mask(flat, 1024).reshape(support0.shape)
    strong8192 = top_mask(flat, 8192).reshape(support0.shape)
    gain[strong8192] = np.clip(gain[strong8192], 0.55, 2.00)
    gain[strong1024] = np.clip(gain[strong1024], 0.70, 1.55)
    gain[strong128] = np.clip(gain[strong128], 0.82, 1.25)

    h1 = leak.leaked_inverse(spec0 * gain)
    h_power0 = np.maximum(
        np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY
    )
    h_power1 = np.maximum(
        np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY
    )
    h1 *= np.sqrt(h_power0 / h_power1)[:, None, None, None].astype(np.float32)
    action = np.sum(np.abs(h1 - h0) ** 2, axis=(1, 2, 3), dtype=np.float64)
    action /= h_power0
    return h1.astype(np.complex64), action.astype(np.float32)


def empty_point_metrics(n: int) -> dict[str, np.ndarray]:
    return {
        "pas": np.zeros(n, dtype=np.float32),
        "pdp": np.zeros(n, dtype=np.float32),
        "error": np.zeros(n, dtype=np.float64),
        "energy": np.zeros(n, dtype=np.float64),
    }


def add_scores(
    output: dict[str, np.ndarray], ids: np.ndarray, prediction: np.ndarray, truth: np.ndarray
) -> None:
    truth_views = metric.spectral_views(truth)
    scores = metric.spectral_scores(prediction, truth_views)
    output["pas"][ids] = scores["pas"]
    output["pdp"][ids] = scores["pdp"]
    output["error"][ids] = np.sum(
        np.abs(prediction - truth) ** 2, axis=(1, 2, 3), dtype=np.float64
    )
    output["energy"][ids] = truth_views["energy"]


def aggregate(values: dict[str, np.ndarray], ids: np.ndarray) -> dict[str, float]:
    pas = float(np.mean(values["pas"][ids]))
    pdp = float(np.mean(values["pdp"][ids]))
    nmse = float(
        np.sum(values["error"][ids], dtype=np.float64)
        / max(np.sum(values["energy"][ids], dtype=np.float64), TINY)
    )
    nmse_score = 1.0 / (1.0 + nmse)
    return {
        "pas": pas,
        "pdp": pdp,
        "global_nmse": nmse,
        "nmse_score": nmse_score,
        "official_local_score": 0.4 * pas + 0.4 * pdp + 0.2 * nmse_score,
    }


def delta_metric(model: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {key: float(model[key] - base[key]) for key in model}


def evaluate_carrier(
    name: str,
    path,
    delta: np.ndarray,
    valid_ids: np.ndarray,
    fold_index: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    base_ad = np.load(path, mmap_mode="r")
    truth_h = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    base_values = empty_point_metrics(2000)
    model_values = empty_point_metrics(2000)
    action = np.zeros(2000, dtype=np.float32)

    for start in range(0, 2000, BATCH):
        end = min(start + BATCH, 2000)
        ids = np.arange(start, end, dtype=np.int64)
        base_h = restore_raw_h(
            np.asarray(base_ad[start:end], dtype=np.complex64), positions[start:end]
        )
        model_h, action[ids] = apply_delta(base_h, delta[start:end])
        truth = np.asarray(truth_h[start:end], dtype=np.complex64)
        add_scores(base_values, ids, base_h, truth)
        add_scores(model_values, ids, model_h, truth)
        print(f"v454 {name} full-H score {end}/2000", flush=True)

    all_ids = np.arange(2000, dtype=np.int64)
    base_valid = aggregate(base_values, valid_ids)
    model_valid = aggregate(model_values, valid_ids)
    base_all = aggregate(base_values, all_ids)
    model_all = aggregate(model_values, all_ids)
    fold_rows = []
    for fold in range(5):
        ids = np.flatnonzero(fold_index == fold).astype(np.int64)
        fold_base = aggregate(base_values, ids)
        fold_model = aggregate(model_values, ids)
        fold_rows.append(
            {
                "fold": fold,
                "count": int(len(ids)),
                "base": fold_base,
                "model": fold_model,
                "delta": delta_metric(fold_model, fold_base),
            }
        )
    result = {
        "carrier": name,
        "source": PATHS.relative(path),
        "valid_1984": {
            "base": base_valid,
            "model": model_valid,
            "delta": delta_metric(model_valid, base_valid),
        },
        "all_2000": {
            "base": base_all,
            "model": model_all,
            "delta": delta_metric(model_all, base_all),
        },
        "positive_folds": int(
            sum(row["delta"]["official_local_score"] > 0.0 for row in fold_rows)
        ),
        "folds": fold_rows,
        "action_nmse": summarize(action[valid_ids]),
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

    delta, valid_ids, fold_index, support = fixed_oof_delta()
    carriers = {}
    details: dict[str, np.ndarray] = {
        "valid_ids": valid_ids.astype(np.int32),
        "fold_index": fold_index,
        "fixed_oof_delta": delta.astype(np.float16),
    }
    for name, path in (
        ("v98like_loo", V98_BASE_AD),
        ("v140eq_crossfit", V140_BASE_AD),
    ):
        result, carrier_details = evaluate_carrier(
            name, path, delta, valid_ids, fold_index
        )
        carriers[name] = result
        details.update(carrier_details)
    np.savez_compressed(DETAILS, **details)

    report = {
        "method": VERSION,
        "status": "analysis_only_official_train_fullh_local_score",
        "primary_question": (
            "What is the standard official-formula local score of the fixed v453 "
            "path residual when evaluated without Sionna test-reference truth?"
        ),
        "metric": {
            "formula": "0.4*PAS + 0.4*PDP + 0.2/(1+global_NMSE)",
            "pas": "2x16x8 BS-array FFT power cosine, averaged over UE/subcarrier/points",
            "pdp": "192-bin IFFT power cosine per 256x4 link, averaged over links/points",
            "truth": "official Round1_Train_Channel.npy only",
            "zero_norm_rule": "nonzero norm product unchanged; zero product denominator=1e-12",
        },
        "adapter": {
            "ridge": RIDGE,
            "strength": STRENGTH,
            "valid_train_count": int(len(valid_ids)),
            "five_fold_oof": True,
            "support_diagnostics": support,
        },
        "carriers": carriers,
        "interpretation_boundary": (
            "v209 has no strictly isomorphic full-H OOF lineage. The v98-like LOO "
            "carrier is the clean early local-score backbone; v140-equivalent is a "
            "later sensitivity carrier previously documented as imperfect."
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
