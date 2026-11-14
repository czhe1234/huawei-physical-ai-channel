#!/usr/bin/env python3
"""Attribute v329 limits to false path existence or the replacement interface."""

from __future__ import annotations

import json

import numpy as np

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v325_path_phase_transport_identifiability as v325
import analyze_v326_complex_path_subspace_transport as v326
import analyze_v328_patch_existence_support_field as v328
import analyze_v329_highprecision_complex_path_transport as v329
from project_paths import PATHS


VERSION_NUMBER = 330
VERSION = "v330_path_transport_oracle_attribution"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"


def evaluate(
    selector: np.ndarray,
    threshold: float,
    rows: dict[str, np.ndarray],
    gain: np.ndarray,
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    train_pos: np.ndarray,
    channel: np.ndarray,
    candidate: dict[str, np.ndarray],
    base: np.ndarray,
    beta: float,
) -> tuple[dict[str, float], dict[str, object]]:
    bank = v329.SelectivePatchBank(
        rows,
        gain,
        fit_ids,
        train_pos[:, :2],
        selector=selector,
        threshold=threshold,
    )
    corrected, diagnostics = v326.corrected_predictions(
        base,
        train_pos[query_ids],
        v326.slice_candidates(candidate, query_ids),
        bank,
        beta,
    )
    metrics = v326.score(corrected, np.asarray(channel[query_ids]))
    return metrics, diagnostics


def main() -> None:
    v329_report = json.loads(v329.REPORT.read_text(encoding="utf-8"))
    gain_data = v325.recover_center_gains()
    rows = v325.matched_rows(gain_data)
    gain = v326.recover_matched_gain_tensors(rows)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    channel = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(component["valid"].any(axis=1)).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    data = np.load(v306.TRAIN_CANDIDATES)
    candidate = {key: data[key] for key in data.files}
    candidate_valid, label = v328.build_labels(candidate)
    fold_rows = []

    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, query_ids)
        threshold = float(v329_report["fold_rows"][fold]["threshold"])
        beta = float(v329_report["fold_rows"][fold]["outer_beta"])
        selector, truth = v329.selector_matrix(
            candidate,
            candidate_valid,
            label,
            fit_ids,
            query_ids,
            train_pos[:, :2],
        )
        selected = selector >= threshold
        true_positive = selected & (truth > 0.0)
        false_positive = selected & (truth <= 0.0)
        base = v326.base_predictions(train_pos[query_ids], fit_ids, train_pos, channel)
        base_metrics = v326.score(base, np.asarray(channel[query_ids]))
        variants = {}
        for name, mask in (
            ("selected_all", selected),
            ("true_positive_only", true_positive),
            ("false_positive_only", false_positive),
        ):
            binary_selector = mask.astype(np.float32)
            metrics, diagnostics = evaluate(
                binary_selector,
                0.5,
                rows,
                gain,
                fit_ids,
                query_ids,
                train_pos,
                channel,
                candidate,
                base,
                beta,
            )
            variants[name] = {
                "metrics": metrics,
                "diagnostics": diagnostics,
                "score_delta": metrics["score"] - base_metrics["score"],
                "pas_delta": metrics["pas"] - base_metrics["pas"],
                "pdp_delta": metrics["pdp"] - base_metrics["pdp"],
                "inv_nmse_delta": metrics["inv_nmse"] - base_metrics["inv_nmse"],
            }
        fold_rows.append(
            {
                "fold": fold,
                "threshold": threshold,
                "selected_pair_count": int(np.sum(selected)),
                "true_positive_pair_count": int(np.sum(true_positive)),
                "false_positive_pair_count": int(np.sum(false_positive)),
                "pair_precision": float(
                    np.sum(true_positive) / max(np.sum(selected), 1)
                ),
                "base": base_metrics,
                "variants": variants,
            }
        )
        print(
            f"v330 fold={fold} precision={fold_rows[-1]['pair_precision']:.3f} "
            f"all={variants['selected_all']['score_delta']:+.5f} "
            f"tp={variants['true_positive_only']['score_delta']:+.5f} "
            f"fp={variants['false_positive_only']['score_delta']:+.5f}",
            flush=True,
        )

    summary = {}
    for name in ("selected_all", "true_positive_only", "false_positive_only"):
        summary[name] = {
            key: float(
                np.mean([row["variants"][name][key] for row in fold_rows])
            )
            for key in ("score_delta", "pas_delta", "pdp_delta", "inv_nmse_delta")
        }
        summary[name]["fold_wins"] = int(
            sum(row["variants"][name]["score_delta"] > 0.0 for row in fold_rows)
        )
    passed = (
        summary["true_positive_only"]["score_delta"] >= 0.01
        and summary["true_positive_only"]["fold_wins"] >= 4
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "diagnostic_truth_policy": (
            "Held-out v305 labels are used only to attribute selected candidates after "
            "v329 thresholds are frozen; no test prediction or deployable selector is trained."
        ),
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "true_positive_only_score_delta": ">=0.01",
            "true_positive_only_fold_wins": ">=4/5",
            "passed": bool(passed),
        },
        "outputs": {"report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "decision": report["decision_rule"]}, indent=2))


if __name__ == "__main__":
    main()
