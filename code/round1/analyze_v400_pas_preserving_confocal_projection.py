#!/usr/bin/env python3
"""Evaluate and deploy a PAS-preserving confocal delay projection."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

import analyze_v205_sionna_config_calibration as calibration
import analyze_v208_strict_historical_replay as v208
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v393_confocal_pointcloud_pdp as v393
import analyze_v395_mass_preserving_confocal_pdp as v395
from generate_v87_easy_field_projection import (
    BS_POS,
    ad_transform,
    beta_subcarriers,
    inverse_ad,
)
from project_paths import PATHS


VERSION_NUMBER = 400
VERSION = "v400_pas_preserving_confocal_projection"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
FIGURE_DIR = REPORT_DIR / "版本四零零完整指标对照图"
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v400_full_metric_oof.npz"
OUTPUT = OUT_DIR / "v400_Round1_Test_Channel.npy"

TRAIN_BASE_AD = PATHS.cache_dir(156) / "train_v140eq_crossfit_base_ad.npy"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
V284_H = PATHS.cache_dir(284) / "v284_Round1_Test_Channel.npy"
V398_TARGETS = PATHS.cache_dir(398) / "v398_relative_confocal_targets.npz"
V284_REPORT = (
    PATHS.version_report_dir(284) / "v284_planar_geometry_residual_field.json"
)
V296_SCORES = (
    PATHS.version_report_dir(296) / "v296_actual_submission_local_scores.csv"
)

VARIANTS = ("base", "delay_only", "sinkhorn", "dual_pas_final", "dual_d_final")
VARIANT_NAMES = {
    "base": "原载体",
    "delay_only": "仅时延缩放",
    "sinkhorn": "角度时延边际守恒",
    "dual_pas_final": "双域迭代-PAS收尾",
    "dual_d_final": "双域迭代-PDP收尾",
}
BATCH = 6
EFFECTIVE_ALPHA = 0.5
SINKHORN_ITERS = 32
DUAL_ITERS = 4
FLOOR = 1e-18
TINY = 1e-30
MIN_SCORE_DELTA = 5e-4
MIN_FOLD_WINS = 4
MIN_LEAKED_DELTA = -2e-4
MIN_PDP_DELTA = 0.0
MIN_NMSE_SCORE_DELTA = -2e-4


def configure_chinese_font() -> None:
    selected = "DejaVu Sans"
    for family in ("Noto Sans CJK JP", "Droid Sans Fallback"):
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family), fallback_to_default=False
            )
            font_manager.fontManager.addfont(path)
            selected = font_manager.FontProperties(fname=path).get_name()
            break
        except ValueError:
            continue
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )


def validate_inputs() -> None:
    required = (
        TRAIN_BASE_AD,
        TRAIN_H,
        V284_H,
        V398_TARGETS,
        V284_REPORT,
        V296_SCORES,
        v393.TARGET_CACHE,
        v393.SCENE_CACHE,
        v393.FEATURE_CACHE,
        PATHS.round1 / "Round1_Train_Pos.npy",
        PATHS.round1 / "Round1_Test_Pos.npy",
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing v400 inputs: {missing}")


def normalize_d(value: np.ndarray) -> np.ndarray:
    result = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return result / np.maximum(result.sum(axis=1, keepdims=True), TINY)


def power_total(ad: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(ad) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64)


def current_d(ad: np.ndarray) -> np.ndarray:
    power = np.abs(ad).astype(np.float32) ** 2
    return normalize_d(
        power.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
    )


def restore_total_power(ad: np.ndarray, target_power: np.ndarray) -> np.ndarray:
    current = np.maximum(power_total(ad), TINY)
    scale = np.sqrt(target_power / current).astype(np.float32)
    return (ad * scale[:, None, None, None, None, None]).astype(np.complex64)


def delay_projection(ad: np.ndarray, target_d: np.ndarray) -> np.ndarray:
    original_power = power_total(ad)
    source_d = current_d(ad)
    gain = np.sqrt(
        np.maximum(target_d, FLOOR) / np.maximum(source_d, FLOOR)
    ).astype(np.float32)
    output = (ad * gain[:, None, None, None, None, :]).astype(np.complex64)
    return restore_total_power(output, original_power)


def sinkhorn_projection(ad: np.ndarray, target_d: np.ndarray) -> np.ndarray:
    original_power = power_total(ad)
    cell_power = np.sum(
        np.abs(ad).astype(np.float32) ** 2,
        axis=(1, 4),
        dtype=np.float64,
    ).astype(np.float64)
    normalized = cell_power / np.maximum(
        cell_power.sum(axis=(1, 2, 3), keepdims=True), TINY
    )
    row_target = normalized.sum(axis=3).reshape(len(ad), -1)
    matrix = np.maximum(normalized.reshape(len(ad), -1, 192), FLOOR)
    matrix /= np.maximum(matrix.sum(axis=(1, 2), keepdims=True), TINY)
    for _ in range(SINKHORN_ITERS):
        matrix *= target_d[:, None, :] / np.maximum(
            matrix.sum(axis=1, keepdims=True), TINY
        )
        matrix *= row_target[:, :, None] / np.maximum(
            matrix.sum(axis=2, keepdims=True), TINY
        )
    matrix *= target_d[:, None, :] / np.maximum(
        matrix.sum(axis=1, keepdims=True), TINY
    )
    matrix *= row_target[:, :, None] / np.maximum(
        matrix.sum(axis=2, keepdims=True), TINY
    )
    desired = matrix.reshape(cell_power.shape)
    gain = np.sqrt(desired / np.maximum(normalized, FLOOR)).astype(np.float32)
    output = (ad * gain[:, None, :, :, None, :]).astype(np.complex64)
    return restore_total_power(output, original_power)


def preserve_frequency_pas(ad: np.ndarray, reference_ad: np.ndarray) -> np.ndarray:
    original_power = power_total(reference_ad)
    spectrum = np.fft.fft(ad, axis=-1).astype(np.complex64)
    reference = np.fft.fft(reference_ad, axis=-1).astype(np.complex64)
    power = np.abs(spectrum).astype(np.float32) ** 2
    reference_power = np.abs(reference).astype(np.float32) ** 2
    group_power = power.sum(axis=(1, 2, 3), keepdims=True, dtype=np.float64)
    reference_distribution = reference_power / np.maximum(
        reference_power.sum(axis=(1, 2, 3), keepdims=True, dtype=np.float64), TINY
    )
    desired = reference_distribution * group_power
    gain = np.sqrt(desired / np.maximum(power, FLOOR)).astype(np.float32)
    gain = np.clip(gain, 0.12, 8.0)
    spectrum *= gain
    output = np.fft.ifft(spectrum, axis=-1).astype(np.complex64)
    return restore_total_power(output, original_power)


def dual_projection(
    ad: np.ndarray, target_d: np.ndarray, *, final_domain: str
) -> np.ndarray:
    output = np.array(ad, dtype=np.complex64, copy=True)
    for _ in range(DUAL_ITERS):
        output = delay_projection(output, target_d)
        output = preserve_frequency_pas(output, ad)
    if final_domain == "d":
        output = delay_projection(output, target_d)
    return output


def raw_ad_from_derotated(
    derotated_ad: np.ndarray,
    positions: np.ndarray,
    beta_s: np.ndarray,
) -> np.ndarray:
    derotated_h = inverse_ad(derotated_ad)
    distance = np.linalg.norm(positions - BS_POS[None, :], axis=1)
    raw_h = derotated_h * np.exp(
        -1j * distance[:, None, None, None] * beta_s[None, :]
    ).astype(np.complex64)
    return ad_transform(raw_h)


def project_variants(ad: np.ndarray, target_d: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "base": np.array(ad, dtype=np.complex64, copy=True),
        "delay_only": delay_projection(ad, target_d),
        "sinkhorn": sinkhorn_projection(ad, target_d),
        "dual_pas_final": dual_projection(ad, target_d, final_domain="pas"),
        "dual_d_final": dual_projection(ad, target_d, final_domain="d"),
    }


def empty_accumulator() -> dict[str, list[np.ndarray] | float]:
    return {
        "pas": [],
        "pdp": [],
        "leaked_sh": [],
        "e2": [],
        "t2": [],
        "action_e2": [],
        "base_e2": [],
        "d_target_l1": [],
        "pas_action_l1": [],
    }


def append_metrics(
    accumulator: dict[str, list[np.ndarray] | float],
    prediction_h: np.ndarray,
    truth_h: np.ndarray,
    base_h: np.ndarray,
    prediction_ad: np.ndarray,
    base_ad: np.ndarray,
    target_d: np.ndarray,
) -> None:
    truth_views = calibration.spectral_views(truth_h)
    scores = calibration.spectral_scores(prediction_h, truth_views)
    e2 = np.sum(
        np.abs(prediction_h - truth_h) ** 2, axis=(1, 2, 3), dtype=np.float64
    )
    t2 = np.sum(np.abs(truth_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
    action_e2 = np.sum(
        np.abs(prediction_h - base_h) ** 2, axis=(1, 2, 3), dtype=np.float64
    )
    base_e2 = np.sum(np.abs(base_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
    base_frequency = np.fft.fft(base_ad, axis=-1)
    pred_frequency = np.fft.fft(prediction_ad, axis=-1)
    base_pas = np.abs(base_frequency).astype(np.float32) ** 2
    pred_pas = np.abs(pred_frequency).astype(np.float32) ** 2
    base_pas /= np.maximum(
        base_pas.sum(axis=(1, 2, 3), keepdims=True, dtype=np.float64), TINY
    )
    pred_pas /= np.maximum(
        pred_pas.sum(axis=(1, 2, 3), keepdims=True, dtype=np.float64), TINY
    )
    accumulator["pas"].append(scores["pas"].astype(np.float64))
    accumulator["pdp"].append(scores["pdp"].astype(np.float64))
    accumulator["leaked_sh"].append(scores["leaked_sh"].astype(np.float64))
    accumulator["e2"].append(e2)
    accumulator["t2"].append(t2)
    accumulator["action_e2"].append(action_e2)
    accumulator["base_e2"].append(base_e2)
    accumulator["d_target_l1"].append(
        np.sum(np.abs(current_d(prediction_ad) - target_d), axis=1)
    )
    accumulator["pas_action_l1"].append(
        np.mean(
            np.sum(np.abs(pred_pas - base_pas), axis=(1, 2, 3)), axis=(1, 2)
        )
    )


def merge_accumulator(
    accumulator: dict[str, list[np.ndarray] | float]
) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate(value).astype(np.float64)
        for key, value in accumulator.items()
    }


def summarize_metrics(values: dict[str, np.ndarray]) -> dict[str, float | int]:
    nmse = float(np.sum(values["e2"]) / max(np.sum(values["t2"]), TINY))
    nmse_score = 1.0 / (1.0 + nmse)
    pas = float(np.mean(values["pas"]))
    pdp = float(np.mean(values["pdp"]))
    leaked = float(np.mean(values["leaked_sh"]))
    point_nmse = values["e2"] / np.maximum(values["t2"], TINY)
    point_score = (
        0.4 * values["leaked_sh"]
        + 0.4 * values["pdp"]
        + 0.2 / (1.0 + point_nmse)
    )
    return {
        "count": int(len(values["pas"])),
        "pas": pas,
        "pdp": pdp,
        "leaked_sh": leaked,
        "global_nmse": nmse,
        "nmse_score": nmse_score,
        "score_conventional": 0.4 * pas + 0.4 * pdp + 0.2 * nmse_score,
        "score_traceback_leaked": 0.4 * leaked + 0.4 * pdp + 0.2 * nmse_score,
        "point_score_mean": float(np.mean(point_score)),
        "action_nmse_mean": float(
            np.mean(values["action_e2"] / np.maximum(values["base_e2"], TINY))
        ),
        "d_target_l1_mean": float(np.mean(values["d_target_l1"])),
        "d_target_l1_max": float(np.max(values["d_target_l1"])),
        "pas_action_l1_mean": float(np.mean(values["pas_action_l1"])),
        "pas_action_l1_max": float(np.max(values["pas_action_l1"])),
    }


def delta_metrics(
    current: dict[str, float | int], base: dict[str, float | int]
) -> dict[str, float]:
    keys = (
        "pas",
        "pdp",
        "leaked_sh",
        "nmse_score",
        "score_conventional",
        "score_traceback_leaked",
        "point_score_mean",
    )
    return {key: float(current[key]) - float(base[key]) for key in keys}


def evaluate_oof() -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]]]:
    target_cache = np.load(v393.TARGET_CACHE)
    truth_sh = v393.v279.normalize(target_cache["sh"].astype(np.float32))
    truth_d = normalize_d(target_cache["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target_cache["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    all_pos = np.vstack([train_pos, test_pos])
    confocal_all, feature_meta = v393.load_or_build_features()
    scene = np.load(v393.SCENE_CACHE)
    scene_feature = np.vstack(
        [scene["train_features"], scene["test_features"]]
    ).astype(np.float32)
    permutation = v393.distance_stratified_permutation(all_pos, 40002)
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    merged_ids = np.concatenate(folds)
    if len(np.unique(merged_ids)) != len(valid_ids) or not np.array_equal(
        np.sort(merged_ids), np.sort(valid_ids)
    ):
        raise ValueError("v400 folds must be mutually exclusive and exhaustive")

    base_ad = np.load(TRAIN_BASE_AD, mmap_mode="r")
    train_h = np.load(TRAIN_H, mmap_mode="r")
    beta_s = beta_subcarriers(140.3, 0.10)
    global_acc = {variant: empty_accumulator() for variant in VARIANTS}
    point_parts: dict[str, dict[str, list[np.ndarray]]] = {
        variant: {key: [] for key in ("ids", "fold", "pas", "pdp", "leaked_sh", "e2", "t2")}
        for variant in VARIANTS
    }
    fold_rows: list[dict[str, object]] = []

    for fold_index, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        fit_anchor = v393.anchor_d(fit_ids, fit_ids, train_pos, truth_sh)
        query_anchor = v393.anchor_d(outer_ids, fit_ids, train_pos, truth_sh)
        residual = v393.fit_predict_residual(
            fit_ids,
            outer_ids,
            fit_anchor,
            query_anchor,
            truth_d,
            all_pos,
            scene_feature,
            confocal_all,
            "full",
            permutation,
            400_000 + fold_index,
        )
        fold_acc = {variant: empty_accumulator() for variant in VARIANTS}
        for start in range(0, len(outer_ids), BATCH):
            end = min(start + BATCH, len(outer_ids))
            ids = outer_ids[start:end]
            raw_base_ad = raw_ad_from_derotated(
                np.asarray(base_ad[ids], dtype=np.complex64),
                train_pos[ids],
                beta_s,
            )
            base_h = inverse_ad(raw_base_ad)
            truth_h = np.asarray(train_h[ids], dtype=np.complex64)
            source_d = current_d(raw_base_ad)
            desired_d = v395.apply_mass_preserving_residual(
                source_d, residual[start:end], EFFECTIVE_ALPHA
            )
            predictions = project_variants(raw_base_ad, desired_d)
            for variant, prediction_ad in predictions.items():
                prediction_h = inverse_ad(prediction_ad)
                append_metrics(
                    fold_acc[variant],
                    prediction_h,
                    truth_h,
                    base_h,
                    prediction_ad,
                    raw_base_ad,
                    desired_d,
                )
                append_metrics(
                    global_acc[variant],
                    prediction_h,
                    truth_h,
                    base_h,
                    prediction_ad,
                    raw_base_ad,
                    desired_d,
                )
        summaries = {
            variant: summarize_metrics(merge_accumulator(fold_acc[variant]))
            for variant in VARIANTS
        }
        base_summary = summaries["base"]
        deltas = {
            variant: delta_metrics(summary, base_summary)
            for variant, summary in summaries.items()
            if variant != "base"
        }
        fold_rows.append(
            {
                "fold": fold_index,
                "query_count": int(len(outer_ids)),
                "fit_count": int(len(fit_ids)),
                "metrics": summaries,
                "deltas_vs_base": deltas,
            }
        )
        print(
            f"v400 fold={fold_index} base={base_summary['score_traceback_leaked']:.6f} "
            f"delay={deltas['delay_only']['score_traceback_leaked']:+.6f} "
            f"sinkhorn={deltas['sinkhorn']['score_traceback_leaked']:+.6f} "
            f"dual={deltas['dual_pas_final']['score_traceback_leaked']:+.6f}",
            flush=True,
        )

    merged = {variant: merge_accumulator(global_acc[variant]) for variant in VARIANTS}
    aggregate = {variant: summarize_metrics(merged[variant]) for variant in VARIANTS}
    aggregate_delta = {
        variant: delta_metrics(summary, aggregate["base"])
        for variant, summary in aggregate.items()
        if variant != "base"
    }
    sinkhorn_fold_delta = np.asarray(
        [
            row["deltas_vs_base"]["sinkhorn"]["score_traceback_leaked"]
            for row in fold_rows
        ],
        dtype=np.float64,
    )
    sinkhorn_delta = aggregate_delta["sinkhorn"]
    passed = bool(
        sinkhorn_delta["score_traceback_leaked"] >= MIN_SCORE_DELTA
        and np.sum(sinkhorn_fold_delta > 0.0) >= MIN_FOLD_WINS
        and sinkhorn_delta["leaked_sh"] >= MIN_LEAKED_DELTA
        and sinkhorn_delta["pdp"] >= MIN_PDP_DELTA
        and sinkhorn_delta["nmse_score"] >= MIN_NMSE_SCORE_DELTA
    )
    np.savez_compressed(
        OOF_CACHE,
        ids=merged_ids.astype(np.int64),
        fold=np.concatenate(
            [np.full(len(ids), index, dtype=np.int8) for index, ids in enumerate(folds)]
        ),
        **{
            f"{variant}_{key}": value.astype(np.float32)
            for variant, fields in merged.items()
            for key, value in fields.items()
            if key in ("pas", "pdp", "leaked_sh", "e2", "t2", "d_target_l1", "pas_action_l1")
        },
    )
    return {
        "feature_meta": feature_meta,
        "fold_count": len(folds),
        "fold_sizes": [int(len(ids)) for ids in folds],
        "mutually_exclusive": True,
        "carrier": (
            "existing v156 official-train LOO v140-equivalent full-H carrier; "
            "query folds are mutually exclusive, but carrier construction predates these folds"
        ),
        "fold_rows": fold_rows,
        "aggregate": aggregate,
        "aggregate_delta_vs_base": aggregate_delta,
        "selection": {
            "primary_variant": "sinkhorn",
            "minimum_score_delta": MIN_SCORE_DELTA,
            "minimum_fold_wins": MIN_FOLD_WINS,
            "minimum_leaked_delta": MIN_LEAKED_DELTA,
            "minimum_pdp_delta": MIN_PDP_DELTA,
            "minimum_nmse_score_delta": MIN_NMSE_SCORE_DELTA,
            "fold_delta": sinkhorn_fold_delta.tolist(),
            "fold_wins": int(np.sum(sinkhorn_fold_delta > 0.0)),
            "passed": passed,
        },
    }, merged


def deploy_sinkhorn() -> dict[str, object]:
    source = np.load(V284_H, mmap_mode="r")
    target_d = normalize_d(np.load(V398_TARGETS)["target_d"].astype(np.float32))
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    diagnostics: dict[str, list[np.ndarray]] = {
        "d_l1": [],
        "pas_action_l1": [],
        "action_nmse": [],
        "power_ratio": [],
    }
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        ad0 = ad_transform(h0)
        ad1 = sinkhorn_projection(ad0, target_d[start:end])
        h1 = inverse_ad(ad1)
        output[start:end] = h1
        output.flush()
        p0 = np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64)
        p1 = np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64)
        diagnostics["d_l1"].append(
            np.sum(np.abs(current_d(ad1) - target_d[start:end]), axis=1)
        )
        diagnostics["action_nmse"].append(
            np.sum(np.abs(h1 - h0) ** 2, axis=(1, 2, 3), dtype=np.float64)
            / np.maximum(p0, TINY)
        )
        diagnostics["power_ratio"].append(p1 / np.maximum(p0, TINY))
        f0 = np.fft.fft(ad0, axis=-1)
        f1 = np.fft.fft(ad1, axis=-1)
        q0 = np.abs(f0).astype(np.float32) ** 2
        q1 = np.abs(f1).astype(np.float32) ** 2
        q0 /= np.maximum(q0.sum(axis=(1, 2, 3), keepdims=True), TINY)
        q1 /= np.maximum(q1.sum(axis=(1, 2, 3), keepdims=True), TINY)
        diagnostics["pas_action_l1"].append(
            np.mean(np.sum(np.abs(q1 - q0), axis=(1, 2, 3)), axis=(1, 2))
        )
        print(f"v400 deployment {end}/{len(source)}", flush=True)
    merged = {key: np.concatenate(value) for key, value in diagnostics.items()}
    return {
        key: {
            "mean": float(np.mean(value)),
            "p90": float(np.quantile(value, 0.90)),
            "max": float(np.max(value)),
        }
        for key, value in merged.items()
    }


def historical_local_scores() -> dict[str, object]:
    v400 = v208.score_prediction(OUTPUT)
    v284_report = json.loads(V284_REPORT.read_text(encoding="utf-8"))
    v284 = v284_report["local_gate"]["score"]
    v212 = None
    with V296_SCORES.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["version"]) == 212:
                v212 = {
                    "score_traceback_leaked": float(row["local_score"]),
                    "pdp": float(row["pdp"]),
                    "leaked_sh": float(row["leaked_sh"]),
                    "nmse_score": float(row["nmse_score"]),
                    "global_nmse": float(row["global_nmse"]),
                }
                break
    if v212 is None:
        raise ValueError("v212 local score missing from v296 history")
    return {
        "v284": v284,
        "v400": v400,
        "v212": v212,
        "v400_delta_vs_v284": {
            key: float(v400[key]) - float(v284[key])
            for key in ("score_traceback_leaked", "pdp", "leaked_sh", "nmse_score")
        },
        "v400_gap_to_v212": {
            key: float(v400[key]) - float(v212[key])
            for key in ("score_traceback_leaked", "pdp", "leaked_sh", "nmse_score")
        },
        "selection_usage": (
            "computed only after OOF method and projector were frozen; not used for training "
            "or selecting v400"
        ),
    }


def plot_oof(analysis: dict[str, object]) -> list[Path]:
    configure_chinese_font()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    aggregate = analysis["aggregate"]
    labels = [VARIANT_NAMES[value] for value in VARIANTS]
    x = np.arange(len(VARIANTS))

    output = FIGURE_DIR / "一、互斥空间折完整指标对照.png"
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fields = (
        ("leaked_sh", "线上口径PAS代理"),
        ("pdp", "PDP"),
        ("nmse_score", "NMSE得分项"),
        ("score_traceback_leaked", "加权综合分"),
    )
    colors = ["#7a8793", "#d97745", "#2a9d8f", "#4878a8", "#8f6bb3"]
    for axis, (field, title) in zip(axes.flat, fields):
        values = [aggregate[value][field] for value in VARIANTS]
        axis.bar(x, values, color=colors)
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.set_title(title)
        low = min(values)
        high = max(values)
        margin = max((high - low) * 0.35, 2e-4)
        axis.set_ylim(low - margin, high + margin)
    figure.suptitle("共焦PDP修正必须同时通过PAS、PDP和NMSE", fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    outputs.append(output)

    output = FIGURE_DIR / "二、五个互斥空间折综合增益.png"
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    width = 0.19
    compared = ("delay_only", "sinkhorn", "dual_pas_final", "dual_d_final")
    fold_x = np.arange(5)
    for offset, variant in enumerate(compared):
        values = [
            row["deltas_vs_base"][variant]["score_traceback_leaked"]
            for row in analysis["fold_rows"]
        ]
        axis.bar(
            fold_x + (offset - 1.5) * width,
            values,
            width,
            label=VARIANT_NAMES[variant],
        )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(fold_x, [f"空间折{value + 1}" for value in fold_x])
    axis.set_ylabel("相对原载体的完整综合分增益")
    axis.set_title("逐折稳定性，不再使用有重叠的五个留出集")
    axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    outputs.append(output)
    return outputs


def plot_local_comparison(local: dict[str, object]) -> Path:
    configure_chinese_font()
    output = FIGURE_DIR / "三、与版本二一二的同口径本地差距.png"
    versions = ("v284", "v400", "v212")
    labels = ("版本二八四", "版本四零零", "版本二一二")
    fields = (
        ("score_traceback_leaked", "综合分"),
        ("pdp", "PDP"),
        ("leaked_sh", "PAS代理"),
        ("nmse_score", "NMSE得分项"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = ["#7a8793", "#2a9d8f", "#c44536"]
    for axis, (field, title) in zip(axes.flat, fields):
        values = [float(local[version][field]) for version in versions]
        axis.bar(labels, values, color=colors)
        axis.set_title(title)
        low = min(values)
        high = max(values)
        margin = max((high - low) * 0.25, 0.002)
        axis.set_ylim(max(0.0, low - margin), min(1.0, high + margin))
        for index, value in enumerate(values):
            axis.text(index, value + margin * 0.12, f"{value:.6f}", ha="center")
    figure.suptitle("冻结方法后才读取历史local replay，与版本二一二对比", fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    validate_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("v400 start mutually exclusive full-metric OOF", flush=True)
    oof, _merged = evaluate_oof()
    figures = plot_oof(oof)
    print("v400 deploy frozen primary sinkhorn projector", flush=True)
    deployment = deploy_sinkhorn()
    print("v400 evaluate historical local replay after freeze", flush=True)
    local = historical_local_scores()
    figures.append(plot_local_comparison(local))

    passed = bool(oof["selection"]["passed"])
    report = {
        "method": VERSION,
        "status": (
            "strict_oof_passed_local_compared_needs_packaging"
            if passed
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "Confocal early-delay information can improve the v284 carrier only when its D "
            "target is imposed through an angle-delay marginal projection that preserves the "
            "carrier PAS, rather than one global gain per delay bin."
        ),
        "parameters": {
            "effective_alpha": EFFECTIVE_ALPHA,
            "active_bins": v393.ACTIVE_BINS,
            "sinkhorn_iterations": SINKHORN_ITERS,
            "dual_iterations": DUAL_ITERS,
            "primary_projector": "sinkhorn angle-delay marginal projection",
            "preserved_angle_marginal": "joint H-V power summed over delay",
        },
        "compliance_boundary": {
            "uses": [
                "official train H and mutually exclusive official train query folds",
                "official train/test coordinates and official Round1 point cloud features",
                "strict official-only v284 test carrier",
                "ExtraTrees confocal residual AI model",
            ],
            "does_not_use": [
                "v212 or synthetic output for training/selection",
                "Sionna/RT test target or path coefficient",
                "hidden online truth or online scores as labels",
            ],
            "historical_local_replay": (
                "read only after OOF method/projector freeze for the user-requested v212 gap"
            ),
        },
        "oof": oof,
        "deployment": deployment,
        "historical_local_comparison": local,
        "outputs": {
            "prediction": PATHS.relative(OUTPUT),
            "oof_cache": PATHS.relative(OOF_CACHE),
            "figures": [PATHS.relative(path) for path in figures],
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
