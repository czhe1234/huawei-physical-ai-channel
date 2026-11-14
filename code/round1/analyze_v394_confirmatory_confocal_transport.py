#!/usr/bin/env python3
"""Confirm half-strength confocal PDP correction on new spatial folds."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import analyze_v279_joint_component_trajectory as v279
import analyze_v393_confocal_pointcloud_pdp as v393
from project_paths import PATHS


VERSION_NUMBER = 394
VERSION = "v394_confirmatory_confocal_transport"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v394_confirmatory_confocal_transport_oof.npz"
DEPLOYMENT_SCALE = 0.5
CIRCULAR_SEAM = 96
FOLD_COUNT = 5
STRATA = 20

MIN_COSINE_LIFT = 0.002
MIN_FULL_VS_CONTEXT = 7.5e-4
MIN_FULL_VS_SHUFFLED = 7.5e-4
MIN_CIRCULAR_EMD_REDUCTION = 0.0
MIN_FOLD_WINS = 4


def confirmatory_folds(
    valid_ids: np.ndarray, train_xy: np.ndarray
) -> list[np.ndarray]:
    nearest = cKDTree(train_xy[valid_ids]).query(
        train_xy[valid_ids], k=2
    )[0][:, 1]
    edges = np.quantile(nearest, np.linspace(0.0, 1.0, STRATA + 1))
    stratum = np.clip(
        np.searchsorted(edges[1:-1], nearest, side="right"), 0, STRATA - 1
    )
    rng = np.random.default_rng(394)
    folds: list[list[int]] = [[] for _ in range(FOLD_COUNT)]
    for value in range(STRATA):
        ids = valid_ids[stratum == value].copy()
        rng.shuffle(ids)
        offset = (2 * value + 1) % FOLD_COUNT
        for index, point_id in enumerate(ids):
            folds[(index + offset) % FOLD_COUNT].append(int(point_id))
    return [np.sort(np.asarray(ids, dtype=np.int64)) for ids in folds]


def circular_emd_rows(
    truth: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    truth = np.roll(truth, -CIRCULAR_SEAM, axis=1)
    prediction = np.roll(prediction, -CIRCULAR_SEAM, axis=1)
    return np.mean(
        np.abs(np.cumsum(truth, axis=1) - np.cumsum(prediction, axis=1)),
        axis=1,
    ).astype(np.float64)


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
    permutation = v393.distance_stratified_permutation(all_pos, 39402)
    folds = confirmatory_folds(valid_ids, train_pos[:, :2])
    modes = ("context", "shuffled", "full")
    fold_rows = []
    oof: dict[str, list[np.ndarray]] = {
        key: [] for key in ("ids", "truth", "anchor", *modes)
    }
    for outer_fold, outer_ids in enumerate(folds):
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = folds[(outer_fold + 2) % len(folds)]
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
                394_000 + 100 * outer_fold + mode_index,
            )
            selected_alpha, selection = v393.choose_alpha(
                truth_d[inner_ids], inner_val_anchor, inner_residual
            )
            effective_alpha = DEPLOYMENT_SCALE * selected_alpha
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
                394_500 + 100 * outer_fold + mode_index,
            )
            prediction = v393.apply_residual(
                outer_anchor, outer_residual, effective_alpha
            )
            predictions[mode] = prediction
            mode_reports[mode] = {
                "selected_alpha": selected_alpha,
                "effective_alpha": effective_alpha,
                "selection": selection,
                "cosine": float(
                    np.mean(v393.row_cos(truth_d[outer_ids], prediction))
                ),
                "circular_emd": float(
                    np.mean(circular_emd_rows(truth_d[outer_ids], prediction))
                ),
            }
        anchor_cosine = float(
            np.mean(v393.row_cos(truth_d[outer_ids], outer_anchor))
        )
        anchor_emd = float(
            np.mean(circular_emd_rows(truth_d[outer_ids], outer_anchor))
        )
        delta = {
            "full_vs_anchor_cosine": mode_reports["full"]["cosine"] - anchor_cosine,
            "full_vs_context_cosine": mode_reports["full"]["cosine"] - mode_reports["context"]["cosine"],
            "full_vs_shuffled_cosine": mode_reports["full"]["cosine"] - mode_reports["shuffled"]["cosine"],
            "full_circular_emd_reduction": (
                anchor_emd - mode_reports["full"]["circular_emd"]
            ) / max(anchor_emd, 1e-30),
        }
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
                "deltas": delta,
            }
        )
        oof["ids"].append(outer_ids)
        oof["truth"].append(truth_d[outer_ids])
        oof["anchor"].append(outer_anchor)
        for mode in modes:
            oof[mode].append(predictions[mode])
        print(
            f"v394 fold={outer_fold} alpha={mode_reports['full']['effective_alpha']:.2f} "
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
    passed = bool(
        summary["full_vs_anchor_cosine"] >= MIN_COSINE_LIFT
        and summary["full_vs_context_cosine"] >= MIN_FULL_VS_CONTEXT
        and summary["full_vs_shuffled_cosine"] >= MIN_FULL_VS_SHUFFLED
        and summary["full_circular_emd_reduction"] >= MIN_CIRCULAR_EMD_REDUCTION
        and summary["fold_wins"] >= MIN_FOLD_WINS
    )
    report = {
        "method": VERSION,
        "status": "confirmatory_confocal_passed" if passed else "analysis_only_no_transfer",
        "primary_hypothesis": (
            "The v393 confocal point-cloud signal survives unseen fold assignment when "
            "the per-bin correction is physically bounded to half strength and delay "
            "transport is measured with a seam opposite the active support."
        ),
        "parameters": {
            "feature_model": "frozen v393 definition",
            "deployment_scale": DEPLOYMENT_SCALE,
            "circular_seam": CIRCULAR_SEAM,
            "fold_seed": 394,
            "folds": "new mutually exclusive isolation-stratified folds",
            "inner_fold_offset": 2,
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
            "passed": passed,
        },
        "compliance_boundary": {
            "uses": [
                "official train H normalized D labels",
                "official coordinates and point cloud",
                "v393 AI feature/model definition",
            ],
            "does_not_use": [
                "v393 fold assignment for confirmation",
                "RT/synthetic channel targets",
                "v212 teacher or hidden/online truth",
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
