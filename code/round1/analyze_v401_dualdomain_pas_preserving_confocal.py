#!/usr/bin/env python3
"""Confirm confocal D correction with evaluator-PAS-preserving dual projections."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v393_confocal_pointcloud_pdp as v393
import analyze_v395_mass_preserving_confocal_pdp as v395
import analyze_v400_pas_preserving_confocal_projection as v400
from generate_v87_easy_field_projection import beta_subcarriers, ad_transform, inverse_ad
from project_paths import PATHS


VERSION_NUMBER = 401
VERSION = "v401_dualdomain_pas_preserving_confocal"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
FIGURE_DIR = REPORT_DIR / "版本四零一双域确认图"
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v401_dualdomain_oof.npz"
OUTPUT = OUT_DIR / "v401_Round1_Test_Channel.npy"

FOLD_COUNT = 5
STRATA = 20
BATCH = 6
MIN_SCORE_DELTA = 5e-4
MIN_FOLD_WINS = 4
MIN_LEAKED_DELTA = 0.0
MIN_PDP_DELTA = 0.0
MIN_NMSE_SCORE_DELTA = -1e-4
TINY = 1e-30


def confirmatory_folds(valid_ids: np.ndarray, train_xy: np.ndarray) -> list[np.ndarray]:
    nearest = cKDTree(train_xy[valid_ids]).query(train_xy[valid_ids], k=2)[0][:, 1]
    edges = np.quantile(nearest, np.linspace(0.0, 1.0, STRATA + 1))
    stratum = np.clip(
        np.searchsorted(edges[1:-1], nearest, side="right"), 0, STRATA - 1
    )
    rng = np.random.default_rng(401)
    folds: list[list[int]] = [[] for _ in range(FOLD_COUNT)]
    for value in range(STRATA):
        ids = valid_ids[stratum == value].copy()
        rng.shuffle(ids)
        offset = (3 * value + 1) % FOLD_COUNT
        for index, point_id in enumerate(ids):
            folds[(index + offset) % FOLD_COUNT].append(int(point_id))
    return [np.sort(np.asarray(ids, dtype=np.int64)) for ids in folds]


def evaluate_oof() -> dict[str, object]:
    target_cache = np.load(v393.TARGET_CACHE)
    truth_sh = v393.v279.normalize(target_cache["sh"].astype(np.float32))
    truth_d = v400.normalize_d(target_cache["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target_cache["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    all_pos = np.vstack([train_pos, test_pos])
    confocal_all, feature_meta = v393.load_or_build_features()
    scene = np.load(v393.SCENE_CACHE)
    scene_feature = np.vstack(
        [scene["train_features"], scene["test_features"]]
    ).astype(np.float32)
    permutation = v393.distance_stratified_permutation(all_pos, 40102)
    folds = confirmatory_folds(valid_ids, train_pos[:, :2])
    merged_ids = np.concatenate(folds)
    if len(np.unique(merged_ids)) != len(valid_ids) or not np.array_equal(
        np.sort(merged_ids), np.sort(valid_ids)
    ):
        raise ValueError("v401 folds must be mutually exclusive and exhaustive")

    base_ad = np.load(v400.TRAIN_BASE_AD, mmap_mode="r")
    train_h = np.load(v400.TRAIN_H, mmap_mode="r")
    beta_s = beta_subcarriers(140.3, 0.10)
    global_acc = {name: v400.empty_accumulator() for name in ("base", "dual_pas")}
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
            401_000 + fold_index,
        )
        fold_acc = {name: v400.empty_accumulator() for name in ("base", "dual_pas")}
        for start in range(0, len(outer_ids), BATCH):
            end = min(start + BATCH, len(outer_ids))
            ids = outer_ids[start:end]
            raw_base_ad = v400.raw_ad_from_derotated(
                np.asarray(base_ad[ids], dtype=np.complex64),
                train_pos[ids],
                beta_s,
            )
            base_h = inverse_ad(raw_base_ad)
            truth_h = np.asarray(train_h[ids], dtype=np.complex64)
            desired_d = v395.apply_mass_preserving_residual(
                v400.current_d(raw_base_ad),
                residual[start:end],
                v400.EFFECTIVE_ALPHA,
            )
            predictions = {
                "base": raw_base_ad,
                "dual_pas": v400.dual_projection(
                    raw_base_ad, desired_d, final_domain="pas"
                ),
            }
            for name, prediction_ad in predictions.items():
                prediction_h = inverse_ad(prediction_ad)
                v400.append_metrics(
                    fold_acc[name],
                    prediction_h,
                    truth_h,
                    base_h,
                    prediction_ad,
                    raw_base_ad,
                    desired_d,
                )
                v400.append_metrics(
                    global_acc[name],
                    prediction_h,
                    truth_h,
                    base_h,
                    prediction_ad,
                    raw_base_ad,
                    desired_d,
                )
        base_summary = v400.summarize_metrics(
            v400.merge_accumulator(fold_acc["base"])
        )
        model_summary = v400.summarize_metrics(
            v400.merge_accumulator(fold_acc["dual_pas"])
        )
        delta = v400.delta_metrics(model_summary, base_summary)
        fold_rows.append(
            {
                "fold": fold_index,
                "query_count": int(len(outer_ids)),
                "fit_count": int(len(fit_ids)),
                "base": base_summary,
                "dual_pas": model_summary,
                "delta": delta,
            }
        )
        print(
            f"v401 fold={fold_index} base={base_summary['score_traceback_leaked']:.6f} "
            f"dual_pas={delta['score_traceback_leaked']:+.6f} "
            f"pdp={delta['pdp']:+.6f} leaked={delta['leaked_sh']:+.6f}",
            flush=True,
        )

    merged = {
        name: v400.merge_accumulator(global_acc[name])
        for name in ("base", "dual_pas")
    }
    aggregate = {
        name: v400.summarize_metrics(value) for name, value in merged.items()
    }
    delta = v400.delta_metrics(aggregate["dual_pas"], aggregate["base"])
    fold_delta = np.asarray(
        [row["delta"]["score_traceback_leaked"] for row in fold_rows],
        dtype=np.float64,
    )
    passed = bool(
        delta["score_traceback_leaked"] >= MIN_SCORE_DELTA
        and np.sum(fold_delta > 0.0) >= MIN_FOLD_WINS
        and delta["leaked_sh"] >= MIN_LEAKED_DELTA
        and delta["pdp"] >= MIN_PDP_DELTA
        and delta["nmse_score"] >= MIN_NMSE_SCORE_DELTA
    )
    np.savez_compressed(
        OOF_CACHE,
        ids=merged_ids.astype(np.int64),
        fold=np.concatenate(
            [np.full(len(ids), index, dtype=np.int8) for index, ids in enumerate(folds)]
        ),
        **{
            f"{name}_{key}": value.astype(np.float32)
            for name, fields in merged.items()
            for key, value in fields.items()
            if key in ("pas", "pdp", "leaked_sh", "e2", "t2", "d_target_l1", "pas_action_l1")
        },
    )
    return {
        "feature_meta": feature_meta,
        "fold_sizes": [int(len(ids)) for ids in folds],
        "mutually_exclusive": True,
        "seed": 401,
        "fold_rows": fold_rows,
        "aggregate": aggregate,
        "delta": delta,
        "selection": {
            "minimum_score_delta": MIN_SCORE_DELTA,
            "minimum_fold_wins": MIN_FOLD_WINS,
            "minimum_leaked_delta": MIN_LEAKED_DELTA,
            "minimum_pdp_delta": MIN_PDP_DELTA,
            "minimum_nmse_score_delta": MIN_NMSE_SCORE_DELTA,
            "fold_delta": fold_delta.tolist(),
            "fold_wins": int(np.sum(fold_delta > 0.0)),
            "passed": passed,
        },
    }


def deploy() -> dict[str, object]:
    source = np.load(v400.V284_H, mmap_mode="r")
    target_d = v400.normalize_d(
        np.load(v400.V398_TARGETS)["target_d"].astype(np.float32)
    )
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    diagnostics = {key: [] for key in ("d_l1", "pas_action_l1", "action_nmse", "power_ratio")}
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        ad0 = ad_transform(h0)
        ad1 = v400.dual_projection(ad0, target_d[start:end], final_domain="pas")
        h1 = inverse_ad(ad1)
        output[start:end] = h1
        output.flush()
        p0 = np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64)
        p1 = np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64)
        diagnostics["d_l1"].append(
            np.sum(np.abs(v400.current_d(ad1) - target_d[start:end]), axis=1)
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
        print(f"v401 deployment {end}/{len(source)}", flush=True)
    merged = {key: np.concatenate(value) for key, value in diagnostics.items()}
    return {
        key: {
            "mean": float(np.mean(value)),
            "p90": float(np.quantile(value, 0.90)),
            "max": float(np.max(value)),
        }
        for key, value in merged.items()
    }


def local_comparison() -> dict[str, object]:
    current = v208.score_prediction(OUTPUT)
    v400_report = json.loads(v400.REPORT.read_text(encoding="utf-8"))
    historical = v400_report["historical_local_comparison"]
    v284 = historical["v284"]
    v212 = historical["v212"]
    keys = ("score_traceback_leaked", "pdp", "leaked_sh", "nmse_score")
    return {
        "v284": v284,
        "v401": current,
        "v212": v212,
        "v401_delta_vs_v284": {
            key: float(current[key]) - float(v284[key]) for key in keys
        },
        "v401_gap_to_v212": {
            key: float(current[key]) - float(v212[key]) for key in keys
        },
        "selection_usage": "read after v401 OOF method freeze only",
    }


def render_figures(oof: dict[str, object], local: dict[str, object]) -> list[str]:
    v400.configure_chinese_font()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    path = FIGURE_DIR / "一、全新互斥空间折确认.png"
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    values = [row["delta"]["score_traceback_leaked"] for row in oof["fold_rows"]]
    axis.bar(
        [f"空间折{index + 1}" for index in range(FOLD_COUNT)],
        values,
        color=["#2a9d8f" if value > 0 else "#c44536" for value in values],
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_ylabel("相对原载体完整综合分增益")
    axis.set_title("双域PAS守恒投影的独立确认")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(PATHS.relative(path))

    path = FIGURE_DIR / "二、与版本二一二同口径本地差距.png"
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    versions = ("v284", "v401", "v212")
    names = ("版本二八四", "版本四零一", "版本二一二")
    fields = (
        ("score_traceback_leaked", "综合分"),
        ("pdp", "PDP"),
        ("leaked_sh", "PAS代理"),
        ("nmse_score", "NMSE得分项"),
    )
    for axis, (field, title) in zip(axes.flat, fields):
        metric = [float(local[name][field]) for name in versions]
        axis.bar(names, metric, color=["#7a8793", "#2a9d8f", "#c44536"])
        low, high = min(metric), max(metric)
        margin = max((high - low) * 0.25, 0.002)
        axis.set_ylim(max(0.0, low - margin), min(1.0, high + margin))
        axis.set_title(title)
        for index, value in enumerate(metric):
            axis.text(index, value + 0.1 * margin, f"{value:.6f}", ha="center")
    figure.suptitle("方法冻结后与版本二一二对比", fontsize=15)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(PATHS.relative(path))
    return paths


def main() -> None:
    v400.validate_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("v401 start independent dual-domain confirmation", flush=True)
    oof = evaluate_oof()
    print("v401 deploy frozen dual-PAS projector", flush=True)
    deployment = deploy()
    print("v401 compare frozen output with v212 local", flush=True)
    local = local_comparison()
    figures = render_figures(oof, local)
    passed = bool(oof["selection"]["passed"])
    local_improves_v284 = bool(
        local["v401_delta_vs_v284"]["score_traceback_leaked"] > 0.0
    )
    report = {
        "method": VERSION,
        "status": (
            "strict_confirmatory_oof_passed_local_negative_not_packaged"
            if passed and not local_improves_v284
            else "strict_confirmatory_oof_passed_needs_packaging"
            if passed and local_improves_v284
            else "analysis_only_no_candidate"
        ),
        "packaging_decision": {
            "official_oof_passed": passed,
            "historical_local_improves_v284": local_improves_v284,
            "packaged": False,
            "reason": (
                "The official OOF mechanism passed, but the frozen output was below v284 on "
                "the user-requested historical local replay and the related v398 action was "
                "already negative online. Preserve the mechanism as evidence; do not submit "
                "this deployment unchanged."
            ),
        },
        "primary_hypothesis": (
            "The confocal D signal survives full-metric validation when alternating delay "
            "projection with an evaluator-PAS magnitude projection and finishing in PAS space."
        ),
        "parameters": {
            "effective_alpha": v400.EFFECTIVE_ALPHA,
            "active_bins": v393.ACTIVE_BINS,
            "dual_iterations": v400.DUAL_ITERS,
            "final_domain": "frequency beam PAS",
            "confirmatory_seed": 401,
        },
        "compliance_boundary": {
            "uses": [
                "official train H and new mutually exclusive official train query folds",
                "official train/test coordinates and official point-cloud confocal features",
                "strict official-only v284 deployment carrier",
                "ExtraTrees confocal residual AI model",
            ],
            "does_not_use": [
                "v212/synthetic output in training or OOF selection",
                "Sionna/RT test target or path coefficient",
                "hidden online truth or online-score labels",
            ],
            "historical_local_replay": "read only after OOF method freeze",
        },
        "oof": oof,
        "deployment": deployment,
        "historical_local_comparison": local,
        "outputs": {
            "prediction": PATHS.relative(OUTPUT),
            "oof_cache": PATHS.relative(OOF_CACHE),
            "figures": figures,
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
