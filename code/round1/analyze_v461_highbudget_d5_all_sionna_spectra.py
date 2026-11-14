#!/usr/bin/env python3
"""Measure whether the v461 high-budget all-interaction spectra add OOF signal."""

from __future__ import annotations

import json

import numpy as np

import analyze_v401_dualdomain_pas_preserving_confocal as v401
import analyze_v432_sionna_link_energy_calibration as v432
import analyze_v447_sionna_linkwise_pdp_shape_calibration as v447
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 461
VERSION = "v461_highbudget_d5_all_sionna_spectra"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}_analysis.json"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)

TRUTH_LINK = PATHS.cache_dir(447) / "official_linkwise_pdp.npy"
OLD_LINK = PATHS.cache_dir(447) / "robust_sionna_linkwise_pdp.npy"
OLD_LINK_QUALITY = PATHS.cache_dir(447) / "sionna_link_quality.npy"
NEW_LINK = OUT_DIR / "train_linkwise_pdp.npy"
OLD_JOINT_OOF = PATHS.cache_dir(434) / "v434_sionna_joint_sh_calibration_oof.npz"
NEW_JOINT = OUT_DIR / "train_joint_sh.npy"

BETAS = np.asarray([0.0, 0.10, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
ALPHAS = np.linspace(0.0, 1.5, 31, dtype=np.float32)
LINK_BATCH = 16


def cosine_grid(
    truth: np.ndarray,
    baseline: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    quality: np.ndarray,
) -> np.ndarray:
    """Return mean cosine over rows for every beta/alpha without materializing candidates."""
    truth = np.asarray(truth, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32)
    old = np.asarray(old, dtype=np.float32)
    new = np.asarray(new, dtype=np.float32)
    quality = np.asarray(quality, dtype=np.float32)
    valid = new.sum(axis=-1) > 0.5
    new = np.where(valid[..., None], new, old)

    tt = np.sum(truth * truth, axis=-1, dtype=np.float64)
    tb = np.sum(truth * baseline, axis=-1, dtype=np.float64)
    bb = np.sum(baseline * baseline, axis=-1, dtype=np.float64)
    output = np.empty((len(BETAS), len(ALPHAS), *tt.shape), dtype=np.float32)
    for beta_index, beta in enumerate(BETAS):
        target = (1.0 - beta) * old + beta * new
        tm = np.sum(truth * target, axis=-1, dtype=np.float64)
        bm = np.sum(baseline * target, axis=-1, dtype=np.float64)
        mm = np.sum(target * target, axis=-1, dtype=np.float64)
        for alpha_index, alpha in enumerate(ALPHAS):
            effective = np.clip(alpha * quality, 0.0, 1.0).astype(np.float64)
            numerator = tb + effective * (tm - tb)
            norm = (
                bb
                + 2.0 * effective * (bm - bb)
                + effective * effective * (mm + bb - 2.0 * bm)
            )
            output[beta_index, alpha_index] = (
                numerator / np.maximum(np.sqrt(tt * norm), 1e-30)
            ).astype(np.float32)
    return output


def summarize_grid(
    scores: np.ndarray,
    baseline: np.ndarray,
    fold_id: np.ndarray,
) -> dict[str, object]:
    mean_grid = scores.mean(axis=-1)
    selected_flat = int(np.argmax(mean_grid))
    beta_index, alpha_index = np.unravel_index(selected_flat, mean_grid.shape)
    selected = scores[beta_index, alpha_index]
    old_index = int(np.argmax(mean_grid[0]))
    old = scores[0, old_index]
    folds = []
    for fold in range(5):
        mask = fold_id == fold
        folds.append(
            {
                "fold": fold,
                "count": int(np.sum(mask)),
                "baseline": float(np.mean(baseline[mask])),
                "old_selected": float(np.mean(old[mask])),
                "selected": float(np.mean(selected[mask])),
                "gain_vs_baseline": float(np.mean(selected[mask] - baseline[mask])),
                "gain_vs_old": float(np.mean(selected[mask] - old[mask])),
            }
        )
    return {
        "selected_beta": float(BETAS[beta_index]),
        "selected_alpha": float(ALPHAS[alpha_index]),
        "baseline_mean": float(np.mean(baseline)),
        "old_best_alpha": float(ALPHAS[old_index]),
        "old_best_mean": float(np.mean(old)),
        "selected_mean": float(np.mean(selected)),
        "gain_vs_baseline": float(np.mean(selected - baseline)),
        "gain_vs_old": float(np.mean(selected - old)),
        "fold_wins_vs_old": int(sum(row["gain_vs_old"] > 0.0 for row in folds)),
        "folds": folds,
        "mean_grid": mean_grid.tolist(),
    }


def analyze_joint(
    folds: list[np.ndarray],
    regimes: np.ndarray,
) -> dict[str, object]:
    cache = np.load(OLD_JOINT_OOF)
    truth = cache["truth"].astype(np.float32)
    baseline = cache["baseline"].astype(np.float32)
    old = cache["calibrated"].astype(np.float32)
    quality = cache["point_quality"].astype(np.float32)
    fold_id = cache["fold_id"].astype(np.int16)
    new_source = np.load(NEW_JOINT, mmap_mode="r").reshape(len(truth), -1)
    new = np.zeros_like(truth)
    all_ids = np.arange(len(truth), dtype=np.int64)
    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, query_ids)
        ratio = v432.fit_ratio(truth, new_source, fit_ids, regimes)
        new[query_ids] = v432.apply_ratio(new_source, query_ids, regimes, ratio)
        if not np.array_equal(np.sort(query_ids), np.flatnonzero(fold_id == fold)):
            raise RuntimeError(f"joint fold mismatch at fold {fold}")
    scores = cosine_grid(
        truth,
        baseline,
        old,
        new,
        quality,
    )
    result = summarize_grid(scores, v432.row_cosine(truth, baseline), fold_id)
    result["new_valid_fraction"] = float(
        np.mean(np.asarray(new_source).sum(axis=1) > 0.5)
    )
    result["raw_new_mean"] = float(
        np.mean(v432.row_cosine(truth, np.asarray(new_source, dtype=np.float32)))
    )
    result["calibrated_new_mean"] = float(np.mean(v432.row_cosine(truth, new)))
    return result


def analyze_link(
    folds: list[np.ndarray],
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, object]:
    truth = np.load(TRUTH_LINK, mmap_mode="r")
    old_source = np.load(OLD_LINK, mmap_mode="r")
    new_source = np.load(NEW_LINK, mmap_mode="r")
    quality = np.load(OLD_LINK_QUALITY, mmap_mode="r")
    all_ids = np.arange(len(truth), dtype=np.int64)
    fold_id = np.full(len(truth), -1, dtype=np.int16)
    scores = np.zeros((len(BETAS), len(ALPHAS), len(truth)), dtype=np.float64)
    baseline_score = np.zeros(len(truth), dtype=np.float64)
    raw_new_score = np.zeros(len(truth), dtype=np.float64)
    calibrated_new_score = np.zeros(len(truth), dtype=np.float64)
    links = truth.shape[1]
    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, query_ids)
        fold_id[query_ids] = fold
        neighbor_ids, weights = v447.neighbor_bank(
            fit_ids, query_ids, positions, blocked, regimes
        )
        for start in range(0, links, LINK_BATCH):
            end = min(start + LINK_BATCH, links)
            baseline = v447.baseline_chunk(
                truth, neighbor_ids, weights, start, end
            )
            old_ratio = v447.fit_ratio_chunk(
                truth, old_source, fit_ids, regimes, start, end
            )
            new_ratio = v447.fit_ratio_chunk(
                truth, new_source, fit_ids, regimes, start, end
            )
            old = v447.apply_ratio_chunk(
                old_source, query_ids, regimes, old_ratio, start, end
            )
            new = v447.apply_ratio_chunk(
                new_source, query_ids, regimes, new_ratio, start, end
            )
            truth_chunk = np.asarray(truth[query_ids, start:end], dtype=np.float32)
            quality_chunk = np.asarray(quality[query_ids, start:end], dtype=np.float32)
            grid = cosine_grid(
                truth_chunk,
                baseline,
                old,
                new,
                quality_chunk,
            )
            scores[:, :, query_ids] += grid.sum(axis=-1)
            baseline_score[query_ids] += v447.link_cosine(
                truth_chunk, baseline
            ).sum(axis=1)
            raw_new_score[query_ids] += v447.link_cosine(
                truth_chunk,
                np.asarray(new_source[query_ids, start:end], dtype=np.float32),
            ).sum(axis=1)
            calibrated_new_score[query_ids] += v447.link_cosine(
                truth_chunk, new
            ).sum(axis=1)
        print(f"v461 link fold {fold + 1}/5 complete", flush=True)
    scores /= links
    baseline_score /= links
    raw_new_score /= links
    calibrated_new_score /= links
    result = summarize_grid(scores, baseline_score, fold_id)
    result["new_valid_link_fraction"] = float(
        np.mean(np.asarray(new_source).sum(axis=-1) > 0.5)
    )
    result["raw_new_mean"] = float(np.mean(raw_new_score))
    result["calibrated_new_mean"] = float(np.mean(calibrated_new_score))
    return result


def main() -> None:
    required = (
        TRUTH_LINK,
        OLD_LINK,
        OLD_LINK_QUALITY,
        NEW_LINK,
        OLD_JOINT_OOF,
        NEW_JOINT,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = geometry["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    all_ids = np.arange(len(positions), dtype=np.int64)
    folds = v401.confirmatory_folds(all_ids, positions[:, :2])

    joint = analyze_joint(folds, regimes)
    link = analyze_link(folds, positions, blocked, regimes)
    pass_gate = bool(
        joint["gain_vs_old"] >= 0.003
        and joint["fold_wins_vs_old"] >= 4
        and link["gain_vs_old"] >= 0.001
        and link["fold_wins_vs_old"] >= 4
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "A high-budget depth-5 all-interaction realization adds path birth/death "
            "information beyond the existing depth-16 and three-frequency references."
        ),
        "configuration": {
            "betas": BETAS.tolist(),
            "alphas": ALPHAS.tolist(),
            "selection": "same five spatial folds as v434/v447",
            "new_reference": "6 GHz, depth 5, 500k samples, all interactions",
        },
        "joint_sh": joint,
        "linkwise_pdp": link,
        "deployment_gate": {
            "pass": pass_gate,
            "requirements": {
                "joint_gain_vs_old": 0.003,
                "joint_fold_wins": 4,
                "link_gain_vs_old": 0.001,
                "link_fold_wins": 4,
            },
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
