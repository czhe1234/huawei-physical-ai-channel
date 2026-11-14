#!/usr/bin/env python3
"""Confirm confocal PDP redistribution with observable active-mass preservation."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import analyze_v279_joint_component_trajectory as v279
import analyze_v393_confocal_pointcloud_pdp as v393
import analyze_v394_confirmatory_confocal_transport as v394
from project_paths import PATHS


VERSION_NUMBER = 395
VERSION = "v395_mass_preserving_confocal_pdp"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v395_mass_preserving_confocal_pdp_oof.npz"
DEPLOYMENT_SCALE = 0.5
CIRCULAR_SEAM = 96
FOLD_COUNT = 5
STRATA = 20

MIN_COSINE_LIFT = 0.002
MIN_FULL_VS_CONTEXT = 7.5e-4
MIN_FULL_VS_SHUFFLED = 7.5e-4
MIN_CIRCULAR_EMD_REDUCTION = 0.0
MIN_FOLD_WINS = 4


def folds_seed395(
    valid_ids: np.ndarray, train_xy: np.ndarray
) -> list[np.ndarray]:
    nearest = cKDTree(train_xy[valid_ids]).query(
        train_xy[valid_ids], k=2
    )[0][:, 1]
    edges = np.quantile(nearest, np.linspace(0.0, 1.0, STRATA + 1))
    stratum = np.clip(
        np.searchsorted(edges[1:-1], nearest, side="right"), 0, STRATA - 1
    )
    rng = np.random.default_rng(395)
    folds: list[list[int]] = [[] for _ in range(FOLD_COUNT)]
    for value in range(STRATA):
        ids = valid_ids[stratum == value].copy()
        rng.shuffle(ids)
        offset = (3 * value + 2) % FOLD_COUNT
        for index, point_id in enumerate(ids):
            folds[(index + offset) % FOLD_COUNT].append(int(point_id))
    return [np.sort(np.asarray(ids, dtype=np.int64)) for ids in folds]


def apply_mass_preserving_residual(
    anchor: np.ndarray, residual: np.ndarray, alpha: float
) -> np.ndarray:
    prediction = v393.apply_residual(anchor, residual, alpha)
    anchor_active = anchor[:, : v393.ACTIVE_BINS].sum(axis=1, keepdims=True)
    prediction_active = prediction[:, : v393.ACTIVE_BINS].sum(
        axis=1, keepdims=True
    )
    prediction_tail = prediction[:, v393.ACTIVE_BINS :].sum(
        axis=1, keepdims=True
    )
    prediction[:, : v393.ACTIVE_BINS] *= anchor_active / np.maximum(
        prediction_active, 1e-30
    )
    prediction[:, v393.ACTIVE_BINS :] *= (1.0 - anchor_active) / np.maximum(
        prediction_tail, 1e-30
    )
    return v393.normalize(prediction)


def choose_alpha(
    truth: np.ndarray, anchor: np.ndarray, residual: np.ndarray
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for alpha in v393.ALPHAS:
        effective = DEPLOYMENT_SCALE * alpha
        prediction = apply_mass_preserving_residual(anchor, residual, effective)
        rows.append(
            {
                "selected_alpha": alpha,
                "effective_alpha": effective,
                "cosine": float(np.mean(v393.row_cos(truth, prediction))),
                "circular_emd": float(
                    np.mean(v394.circular_emd_rows(truth, prediction))
                ),
            }
        )
    selected = max(rows, key=lambda row: (row["cosine"], -row["circular_emd"]))
    return float(selected["effective_alpha"]), rows


def main() -> None:
    confocal_all, feature_meta = v393.load_or_build_features()
    target = np.load(v393.TARGET_CACHE)
    truth_sh = v279.normalize(target["sh"].astype(np.float32))
    truth_d = v393.normalize(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    all_pos = np.vstack([train_pos, test_pos])
    scene = np.load(v393.SCENE_CACHE)
    scene_feature = np.vstack(
        [scene["train_features"], scene["test_features"]]
    ).astype(np.float32)
    permutation = v393.distance_stratified_permutation(all_pos, 39502)
    folds = folds_seed395(valid_ids, train_pos[:, :2])
    modes = ("context", "shuffled", "full")
    fold_rows = []
    oof: dict[str, list[np.ndarray]] = {
        key: [] for key in ("ids", "truth", "anchor", *modes)
    }
    for outer_fold, outer_ids in enumerate(folds):
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = folds[(outer_fold + 3) % len(folds)]
        inner_fit = np.setdiff1d(outer_fit, inner_ids)
        inner_fit_anchor = v393.anchor_d(
            inner_fit, inner_fit, train_pos, truth_sh
        )
        inner_val_anchor = v393.anchor_d(
            inner_ids, inner_fit, train_pos, truth_sh
        )
        outer_fit_anchor = v393.anchor_d(
            outer_fit, outer_fit, train_pos, truth_sh
        )
        outer_anchor = v393.anchor_d(
            outer_ids, outer_fit, train_pos, truth_sh
        )
        predictions = {}
        mode_reports = {}
        for mode_index, mode in enumerate(modes):
            inner_residual = v393.fit_predict_residual(
                inner_fit,
                inner_ids,
                inner_fit_anchor,
                inner_val_anchor,
                truth_d,
                all_pos,
                scene_feature,
                confocal_all,
                mode,
                permutation,
                395_000 + 100 * outer_fold + mode_index,
            )
            effective_alpha, selection = choose_alpha(
                truth_d[inner_ids], inner_val_anchor, inner_residual
            )
            outer_residual = v393.fit_predict_residual(
                outer_fit,
                outer_ids,
                outer_fit_anchor,
                outer_anchor,
                truth_d,
                all_pos,
                scene_feature,
                confocal_all,
                mode,
                permutation,
                395_500 + 100 * outer_fold + mode_index,
            )
            prediction = apply_mass_preserving_residual(
                outer_anchor, outer_residual, effective_alpha
            )
            predictions[mode] = prediction
            mode_reports[mode] = {
                "effective_alpha": effective_alpha,
                "selection": selection,
                "cosine": float(
                    np.mean(v393.row_cos(truth_d[outer_ids], prediction))
                ),
                "circular_emd": float(
                    np.mean(
                        v394.circular_emd_rows(truth_d[outer_ids], prediction)
                    )
                ),
            }
        anchor_cosine = float(
            np.mean(v393.row_cos(truth_d[outer_ids], outer_anchor))
        )
        anchor_emd = float(
            np.mean(v394.circular_emd_rows(truth_d[outer_ids], outer_anchor))
        )
        delta = {
            "full_vs_anchor_cosine": mode_reports["full"]["cosine"] - anchor_cosine,
            "full_vs_context_cosine": mode_reports["full"]["cosine"] - mode_reports["context"]["cosine"],
            "full_vs_shuffled_cosine": mode_reports["full"]["cosine"] - mode_reports["shuffled"]["cosine"],
            "full_circular_emd_reduction": (
                anchor_emd - mode_reports["full"]["circular_emd"]
            ) / max(anchor_emd, 1e-30),
        }
        active_mass_error = float(
            np.max(
                np.abs(
                    predictions["full"][:, : v393.ACTIVE_BINS].sum(axis=1)
                    - outer_anchor[:, : v393.ACTIVE_BINS].sum(axis=1)
                )
            )
        )
        fold_rows.append(
            {
                "fold": outer_fold,
                "query_count": int(len(outer_ids)),
                "inner_query_count": int(len(inner_ids)),
                "anchor": {
                    "cosine": anchor_cosine,
                    "circular_emd": anchor_emd,
                },
                "modes": mode_reports,
                "active_mass_max_abs_error": active_mass_error,
                "deltas": delta,
            }
        )
        oof["ids"].append(outer_ids)
        oof["truth"].append(truth_d[outer_ids])
        oof["anchor"].append(outer_anchor)
        for mode in modes:
            oof[mode].append(predictions[mode])
        print(
            f"v395 fold={outer_fold} alpha={mode_reports['full']['effective_alpha']:.2f} "
            f"delta={delta['full_vs_anchor_cosine']:+.6f} "
            f"vs_context={delta['full_vs_context_cosine']:+.6f} "
            f"vs_shuffle={delta['full_vs_shuffled_cosine']:+.6f} "
            f"emd={delta['full_circular_emd_reduction']:+.3%}",
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
            and row["deltas"]["full_circular_emd_reduction"] >= 0
            for row in fold_rows
        )
    )
    summary["active_mass_max_abs_error"] = float(
        max(row["active_mass_max_abs_error"] for row in fold_rows)
    )
    passed = bool(
        summary["full_vs_anchor_cosine"] >= MIN_COSINE_LIFT
        and summary["full_vs_context_cosine"] >= MIN_FULL_VS_CONTEXT
        and summary["full_vs_shuffled_cosine"] >= MIN_FULL_VS_SHUFFLED
        and summary["full_circular_emd_reduction"] >= MIN_CIRCULAR_EMD_REDUCTION
        and summary["fold_wins"] >= MIN_FOLD_WINS
        and summary["active_mass_max_abs_error"] <= 2e-6
    )
    report = {
        "method": VERSION,
        "status": "mass_preserving_confocal_passed" if passed else "analysis_only_no_transfer",
        "primary_hypothesis": (
            "Confocal point-cloud features should redistribute path mass within the "
            "physically active delay region while the stronger anchor retains control "
            "of active-versus-periodic-tail total mass."
        ),
        "parameters": {
            "feature_model": "frozen v393 definition",
            "deployment_scale": DEPLOYMENT_SCALE,
            "preserved_groups": [[0, v393.ACTIVE_BINS], [v393.ACTIVE_BINS, 192]],
            "circular_seam": CIRCULAR_SEAM,
            "fold_seed": 395,
            "inner_fold_offset": 3,
            "alphas_before_scale": list(v393.ALPHAS),
        },
        "feature_cache": feature_meta,
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "full_vs_anchor_cosine": f">={MIN_COSINE_LIFT}",
            "full_vs_context_cosine": f">={MIN_FULL_VS_CONTEXT}",
            "full_vs_shuffled_cosine": f">={MIN_FULL_VS_SHUFFLED}",
            "full_circular_emd_reduction": f">={MIN_CIRCULAR_EMD_REDUCTION}",
            "joint_fold_wins": f">={MIN_FOLD_WINS}/5",
            "active_mass_max_abs_error": "<=2e-6",
            "passed": passed,
        },
        "compliance_boundary": {
            "uses": [
                "official train H normalized D labels",
                "official coordinates and point cloud",
                "AI confocal residual and observable anchor group mass",
            ],
            "does_not_use": [
                "RT/synthetic target channels",
                "v212 teacher",
                "hidden, local-replay, or online truth",
            ],
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
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
