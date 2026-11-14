#!/usr/bin/env python3
"""Apply complex path transport only to high-precision existence candidates."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import analyze_v273_planar_barrier_kernel as v273
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v325_path_phase_transport_identifiability as v325
import analyze_v326_complex_path_subspace_transport as v326
import analyze_v328_patch_existence_support_field as v328
from project_paths import PATHS


VERSION_NUMBER = 329
VERSION = "v329_highprecision_complex_path_transport"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUT_DIR / "v329_Round1_Test_Channel.npy"
REPORT = REPORT_DIR / f"{VERSION}.json"


class SelectivePatchBank(v326.PatchBank):
    def __init__(self, *args, selector: np.ndarray, threshold: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.selector = selector
        self.threshold = threshold

    def candidates(
        self,
        query_xy: np.ndarray,
        candidate: dict[str, np.ndarray],
        query_row: int,
    ) -> list[tuple[float, int, int, float]]:
        valid = np.flatnonzero(
            (candidate["score"][query_row] > 0.0)
            & (self.selector[query_row] >= self.threshold)
        )
        selected: list[tuple[float, int, int, float]] = []
        used_patch: set[int] = set()
        for candidate_id in valid:
            patch_id = int(candidate["patch_id"][query_row, candidate_id])
            if patch_id in used_patch or patch_id not in self.patch:
                continue
            bank_rows, tree = self.patch[patch_id]
            distance, local = tree.query(query_xy, k=1)
            if distance > v326.MAX_PAIR_DISTANCE:
                continue
            bank_row = int(bank_rows[int(local)])
            score = float(self.selector[query_row, candidate_id])
            selected.append((score, int(candidate_id), bank_row, float(distance)))
            used_patch.add(patch_id)
        selected.sort(reverse=True)
        return selected[: v326.MAX_PATHS]


def selector_matrix(
    candidate: dict[str, np.ndarray],
    valid: np.ndarray,
    label: np.ndarray,
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point, candidate_id, score, truth = v328.predict(
        candidate, valid, label, fit_ids, query_ids, xy
    )
    lookup = {int(point_id): row for row, point_id in enumerate(query_ids)}
    matrix = np.zeros((len(query_ids), candidate["score"].shape[1]), dtype=np.float32)
    truth_matrix = np.zeros_like(matrix)
    for source_point, source_candidate, value, target in zip(
        point, candidate_id, score, truth
    ):
        row = lookup[int(source_point)]
        matrix[row, int(source_candidate)] = value
        truth_matrix[row, int(source_candidate)] = target
    return matrix, truth_matrix


def cross_dataset_selector(
    train_candidate: dict[str, np.ndarray],
    train_valid: np.ndarray,
    label: np.ndarray,
    fit_ids: np.ndarray,
    train_xy: np.ndarray,
    query_candidate: dict[str, np.ndarray],
    query_xy: np.ndarray,
) -> np.ndarray:
    fit_patch = v328.patch_rows(train_candidate, train_valid, fit_ids)
    tree_cache: dict[int, tuple[np.ndarray, np.ndarray, cKDTree]] = {}
    output = np.zeros(query_candidate["score"].shape, dtype=np.float32)
    for query_row in range(len(query_xy)):
        for candidate_id in np.flatnonzero(query_candidate["score"][query_row] > 0.0):
            patch_id = int(query_candidate["patch_id"][query_row, candidate_id])
            if patch_id not in fit_patch:
                continue
            if patch_id not in tree_cache:
                fit_point, fit_candidate = fit_patch[patch_id]
                tree_cache[patch_id] = (
                    fit_point,
                    fit_candidate,
                    cKDTree(train_xy[fit_point]),
                )
            fit_point, fit_candidate, tree = tree_cache[patch_id]
            count = min(v328.K_SUPPORT, len(fit_point))
            distance, local = tree.query(query_xy[query_row], k=count)
            distance = np.atleast_1d(distance)
            local = np.atleast_1d(local)
            weight = 1.0 / np.maximum(distance, v328.DISTANCE_FLOOR) ** 2
            neighbor = (label[fit_point[local], fit_candidate[local]] > 0.0).astype(float)
            output[query_row, candidate_id] = float(np.sum(weight * neighbor) / np.sum(weight))
    return output


def main() -> None:
    gain_data = v325.recover_center_gains()
    rows = v325.matched_rows(gain_data)
    gain = v326.recover_matched_gain_tensors(rows)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    channel = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(component["valid"].any(axis=1)).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    train_data = np.load(v306.TRAIN_CANDIDATES)
    train_candidate = {key: train_data[key] for key in train_data.files}
    candidate_valid, label = v328.build_labels(train_candidate)
    fold_rows = []
    selected_alphas = []
    selected_thresholds = []

    for fold, outer_ids in enumerate(folds):
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = v273.matched_subset(
            outer_fit,
            train_pos[:, :2],
            test_pos[:, :2],
            min(240, max(len(outer_fit) // 5, 1)),
            32900 + fold,
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)
        inner_selector, inner_truth = selector_matrix(
            train_candidate,
            candidate_valid,
            label,
            inner_fit,
            inner_ids,
            train_pos[:, :2],
        )
        active_inner = candidate_valid[inner_ids]
        high_precision = v328.precision_recall_at_target(
            inner_selector[active_inner], inner_truth[active_inner], v328.PRECISION_TARGET
        )
        threshold = float(high_precision["score_threshold"])
        selected_thresholds.append(threshold)
        inner_beta = v326.estimate_beta(rows, inner_fit, train_pos[:, :2])
        inner_base = v326.base_predictions(
            train_pos[inner_ids], inner_fit, train_pos, channel
        )
        inner_bank = SelectivePatchBank(
            rows,
            gain,
            inner_fit,
            train_pos[:, :2],
            selector=inner_selector,
            threshold=threshold,
        )
        inner_corrected, inner_diagnostics = v326.corrected_predictions(
            inner_base,
            train_pos[inner_ids],
            v326.slice_candidates(train_candidate, inner_ids),
            inner_bank,
            inner_beta,
        )
        alpha, alpha_rows = v326.choose_alpha(
            inner_base, inner_corrected, np.asarray(channel[inner_ids])
        )
        selected_alphas.append(alpha)

        outer_selector, _outer_truth = selector_matrix(
            train_candidate,
            candidate_valid,
            label,
            outer_fit,
            outer_ids,
            train_pos[:, :2],
        )
        outer_beta = v326.estimate_beta(rows, outer_fit, train_pos[:, :2])
        outer_base = v326.base_predictions(
            train_pos[outer_ids], outer_fit, train_pos, channel
        )
        outer_bank = SelectivePatchBank(
            rows,
            gain,
            outer_fit,
            train_pos[:, :2],
            selector=outer_selector,
            threshold=threshold,
        )
        outer_corrected, outer_diagnostics = v326.corrected_predictions(
            outer_base,
            train_pos[outer_ids],
            v326.slice_candidates(train_candidate, outer_ids),
            outer_bank,
            outer_beta,
        )
        truth = np.asarray(channel[outer_ids])
        base_metrics = v326.score(outer_base, truth)
        prediction_metrics = v326.score(
            v326.blend(outer_base, outer_corrected, alpha), truth
        )
        fold_rows.append(
            {
                "fold": fold,
                "threshold": threshold,
                "high_precision_inner": high_precision,
                "selected_alpha": alpha,
                "inner_beta": inner_beta,
                "outer_beta": outer_beta,
                "inner_alpha_rows": alpha_rows,
                "inner_diagnostics": inner_diagnostics,
                "outer_diagnostics": outer_diagnostics,
                "base": base_metrics,
                "prediction": prediction_metrics,
                "score_delta": prediction_metrics["score"] - base_metrics["score"],
                "pas_delta": prediction_metrics["pas"] - base_metrics["pas"],
                "pdp_delta": prediction_metrics["pdp"] - base_metrics["pdp"],
                "inv_nmse_delta": prediction_metrics["inv_nmse"]
                - base_metrics["inv_nmse"],
            }
        )
        print(
            f"v329 fold={fold} threshold={threshold:.3f} alpha={alpha:.2f} "
            f"active={outer_diagnostics['active_fraction']:.3f} "
            f"score={fold_rows[-1]['score_delta']:+.6f}",
            flush=True,
        )

    score_delta = float(np.mean([row["score_delta"] for row in fold_rows]))
    pas_delta = float(np.mean([row["pas_delta"] for row in fold_rows]))
    pdp_delta = float(np.mean([row["pdp_delta"] for row in fold_rows]))
    inv_nmse_delta = float(np.mean([row["inv_nmse_delta"] for row in fold_rows]))
    fold_wins = int(sum(row["score_delta"] > 0.0 for row in fold_rows))
    passed = (
        score_delta >= v326.MIN_SCORE_DELTA
        and fold_wins >= 4
        and pas_delta >= -v326.MAX_SPECTRAL_DEGRADATION
        and pdp_delta >= -v326.MAX_SPECTRAL_DEGRADATION
    )

    deployment = None
    if passed:
        final_alpha = float(np.median(selected_alphas))
        final_threshold = float(np.median(selected_thresholds))
        final_beta = v326.estimate_beta(rows, valid_ids, train_pos[:, :2])
        test_data = np.load(v306.TEST_CANDIDATES)
        test_candidate = {key: test_data[key] for key in test_data.files}
        test_selector = cross_dataset_selector(
            train_candidate,
            candidate_valid,
            label,
            valid_ids,
            train_pos[:, :2],
            test_candidate,
            test_pos[:, :2],
        )
        test_base = np.load(v326.BASE_TEST, mmap_mode="r")
        bank = SelectivePatchBank(
            rows,
            gain,
            valid_ids,
            train_pos[:, :2],
            selector=test_selector,
            threshold=final_threshold,
        )
        corrected, diagnostics = v326.corrected_predictions(
            np.asarray(test_base), test_pos, test_candidate, bank, final_beta
        )
        prediction = v326.blend(np.asarray(test_base), corrected, final_alpha)
        np.save(OUTPUT, prediction.astype(np.complex64))
        deployment = {
            "alpha": final_alpha,
            "threshold": final_threshold,
            "beta": final_beta,
            "prediction": PATHS.relative(OUTPUT),
            "shape": list(prediction.shape),
            "dtype": str(prediction.dtype),
            "finite": bool(np.isfinite(prediction).all()),
            "diagnostics": diagnostics,
        }

    report = {
        "method": VERSION,
        "status": "official_selection_passed" if passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The v328 high-precision path-existence field can safely release v327 "
            "path-specific complex phase transport in a full-H predictor."
        ),
        "fold_rows": fold_rows,
        "oof": {
            "score_delta": score_delta,
            "pas_delta": pas_delta,
            "pdp_delta": pdp_delta,
            "inv_nmse_delta": inv_nmse_delta,
            "fold_wins": fold_wins,
        },
        "decision_rule": {
            "score_delta": f">={v326.MIN_SCORE_DELTA}",
            "fold_wins": ">=4/5",
            "pas_delta": f">=-{v326.MAX_SPECTRAL_DEGRADATION}",
            "pdp_delta": f">=-{v326.MAX_SPECTRAL_DEGRADATION}",
            "passed": bool(passed),
        },
        "deployment": deployment,
        "outputs": {
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"oof": report["oof"], "decision": report["decision_rule"]}, indent=2))


if __name__ == "__main__":
    main()
