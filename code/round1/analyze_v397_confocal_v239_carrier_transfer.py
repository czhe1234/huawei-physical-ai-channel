#!/usr/bin/env python3
"""Audit confocal relative D actions on strict v239-like cross-fit carriers."""

from __future__ import annotations

import json

import numpy as np

import analyze_v279_joint_component_trajectory as v279
import analyze_v281_v239_crossfit_pdp_current as v281
import analyze_v393_confocal_pointcloud_pdp as v393
import analyze_v394_confirmatory_confocal_transport as v394
import analyze_v395_mass_preserving_confocal_pdp as v395
from project_paths import PATHS


VERSION_NUMBER = 397
VERSION = "v397_confocal_v239_carrier_transfer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v397_confocal_v239_carrier_transfer_oof.npz"
V281_CACHE = PATHS.cache_dir(281) / "v281_v239_crossfit_pdp_current.npz"
EFFECTIVE_ALPHA = 0.5

MIN_COSINE_LIFT = 0.002
MIN_FULL_VS_CONTEXT = 7.5e-4
MIN_FULL_VS_SHUFFLED = 7.5e-4
MIN_CIRCULAR_EMD_REDUCTION = 0.0
MIN_FOLD_WINS = 4


def main() -> None:
    for path in (V281_CACHE, v393.TARGET_CACHE, v393.FEATURE_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    carrier_cache = np.load(V281_CACHE)
    confocal_all, feature_meta = v393.load_or_build_features()
    target = np.load(v393.TARGET_CACHE)
    truth_sh = v279.normalize(target["sh"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    all_pos = np.vstack([train_pos, test_pos])
    scene = np.load(v393.SCENE_CACHE)
    scene_feature = np.vstack(
        [scene["train_features"], scene["test_features"]]
    ).astype(np.float32)
    permutation = v393.distance_stratified_permutation(all_pos, 39702)
    modes = ("context", "shuffled", "full")
    fold_rows = []
    oof: dict[str, list[np.ndarray]] = {
        key: [] for key in ("fold", "ids", "truth", "carrier", *modes)
    }
    for fold in range(5):
        rows = np.flatnonzero(carrier_cache["fold"] == fold)
        outer_ids = carrier_cache["ids"][rows].astype(np.int64)
        outer_fit = np.setdiff1d(valid_ids, np.unique(outer_ids))
        carrier = v393.normalize(carrier_cache["current_d"][rows].astype(np.float32))
        truth = v393.normalize(carrier_cache["truth_d"][rows].astype(np.float32))
        fit_anchor = v393.anchor_d(outer_fit, outer_fit, train_pos, truth_sh)
        query_anchor = v393.anchor_d(outer_ids, outer_fit, train_pos, truth_sh)
        predictions = {}
        mode_reports = {}
        for mode_index, mode in enumerate(modes):
            residual = v393.fit_predict_residual(
                outer_fit,
                outer_ids,
                fit_anchor,
                query_anchor,
                v393.normalize(target["d"].astype(np.float32)),
                all_pos,
                scene_feature,
                confocal_all,
                mode,
                permutation,
                397_000 + 100 * fold + mode_index,
            )
            prediction = v395.apply_mass_preserving_residual(
                carrier, residual, EFFECTIVE_ALPHA
            )
            predictions[mode] = prediction
            mode_reports[mode] = {
                "cosine": float(np.mean(v393.row_cos(truth, prediction))),
                "circular_emd": float(
                    np.mean(v394.circular_emd_rows(truth, prediction))
                ),
            }
        carrier_cosine = float(np.mean(v393.row_cos(truth, carrier)))
        carrier_emd = float(np.mean(v394.circular_emd_rows(truth, carrier)))
        delta = {
            "full_vs_carrier_cosine": mode_reports["full"]["cosine"] - carrier_cosine,
            "full_vs_context_cosine": mode_reports["full"]["cosine"] - mode_reports["context"]["cosine"],
            "full_vs_shuffled_cosine": mode_reports["full"]["cosine"] - mode_reports["shuffled"]["cosine"],
            "full_circular_emd_reduction": (
                carrier_emd - mode_reports["full"]["circular_emd"]
            ) / max(carrier_emd, 1e-30),
        }
        active_mass_error = float(
            np.max(
                np.abs(
                    predictions["full"][:, : v393.ACTIVE_BINS].sum(axis=1)
                    - carrier[:, : v393.ACTIVE_BINS].sum(axis=1)
                )
            )
        )
        fold_rows.append(
            {
                "fold": fold,
                "query_count": int(len(outer_ids)),
                "fit_count": int(len(outer_fit)),
                "carrier": {
                    "cosine": carrier_cosine,
                    "circular_emd": carrier_emd,
                },
                "modes": mode_reports,
                "active_mass_max_abs_error": active_mass_error,
                "deltas": delta,
            }
        )
        oof["fold"].append(np.full(len(rows), fold, dtype=np.int8))
        oof["ids"].append(outer_ids)
        oof["truth"].append(truth)
        oof["carrier"].append(carrier)
        for mode in modes:
            oof[mode].append(predictions[mode])
        print(
            f"v397 fold={fold} delta={delta['full_vs_carrier_cosine']:+.6f} "
            f"vs_context={delta['full_vs_context_cosine']:+.6f} "
            f"vs_shuffle={delta['full_vs_shuffled_cosine']:+.6f} "
            f"emd={delta['full_circular_emd_reduction']:+.3%}",
            flush=True,
        )
    merged = {key: np.concatenate(value) for key, value in oof.items()}
    np.savez_compressed(OOF_CACHE, **merged)
    summary = {
        key: float(np.mean([row["deltas"][key] for row in fold_rows]))
        for key in fold_rows[0]["deltas"]
    }
    summary["fold_wins"] = int(
        sum(
            row["deltas"]["full_vs_carrier_cosine"] > 0
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
        summary["full_vs_carrier_cosine"] >= MIN_COSINE_LIFT
        and summary["full_vs_context_cosine"] >= MIN_FULL_VS_CONTEXT
        and summary["full_vs_shuffled_cosine"] >= MIN_FULL_VS_SHUFFLED
        and summary["full_circular_emd_reduction"] >= MIN_CIRCULAR_EMD_REDUCTION
        and summary["fold_wins"] >= MIN_FOLD_WINS
        and summary["active_mass_max_abs_error"] <= 2e-6
    )
    report = {
        "method": VERSION,
        "status": "carrier_transfer_passed" if passed else "analysis_only_no_transfer",
        "primary_hypothesis": (
            "The confocal model's relative within-active-delay action transfers from its "
            "KNN training anchor to the isomorphic v239-like cross-fit current used by the "
            "strict v284 deployment carrier."
        ),
        "parameters": {
            "carrier": "frozen v281 v239-like cross-fit current",
            "effective_alpha": EFFECTIVE_ALPHA,
            "active_mass_preservation": True,
            "confocal_model": "frozen v393 definition",
            "carrier_folds": list(v281.OUTER_SEEDS),
        },
        "feature_cache": feature_meta,
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "full_vs_carrier_cosine": f">={MIN_COSINE_LIFT}",
            "full_vs_context_cosine": f">={MIN_FULL_VS_CONTEXT}",
            "full_vs_shuffled_cosine": f">={MIN_FULL_VS_SHUFFLED}",
            "full_circular_emd_reduction": f">={MIN_CIRCULAR_EMD_REDUCTION}",
            "joint_fold_wins": f">={MIN_FOLD_WINS}/5",
            "active_mass_max_abs_error": "<=2e-6",
            "passed": passed,
        },
        "compliance_boundary": {
            "uses": [
                "official train H D labels",
                "official coordinates and point cloud",
                "strict fit-only v239-like official cross-fit current",
            ],
            "does_not_use": [
                "RT/synthetic/v212/hidden/online truth",
                "v396 test target for training or selection",
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
