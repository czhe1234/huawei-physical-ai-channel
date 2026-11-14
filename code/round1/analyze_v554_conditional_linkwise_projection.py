#!/usr/bin/env python3
"""Audit the v554 linkwise-PDP residual under fixed global delay mass."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import analyze_v208_strict_historical_replay as replay
import analyze_v401_dualdomain_pas_preserving_confocal as v401
import analyze_v447_sionna_linkwise_pdp_shape_calibration as v447
import analyze_v554_multiseed_linkwise_pdp_calibration as calibration
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
import generate_v554_multiseed_linkwise_pdp_statistics as source
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 554
VERSION = "v554_conditional_linkwise_projection"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER) / "conditional_linkwise_projection"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
NEW_CALIBRATED_OOF = OUT_DIR / "new_calibrated_linkwise_pdp_oof.npy"
NEW_CALIBRATED_DONE = OUT_DIR / "new_calibrated_linkwise_pdp_oof.done"
CONDITIONAL_OOF = OUT_DIR / "conditional_linkwise_pdp_oof.npz"
TEST_DESIRED = OUT_DIR / "v554_conditional_test_linkwise_pdp.npy"
TEST_OUTPUT = OUT_DIR / "v554_conditional_probe_Round1_Test_Channel.npy"

BASE = PATHS.cache_dir(549) / "v549_Round1_Test_Channel.npy"
FULL_ACTION = calibration.TEST_ACTION
TRUTH = v447.TRUTH_CACHE
OLD_OOF = calibration.OLD_CALIBRATED_OOF
CHECKPOINT = 24
LINK_BATCH = 16
SINKHORN_ITERATIONS = 40
PROJECTION_CYCLES = 1
BATCH = 2
FLOOR = 1e-12


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def summarize(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "mean": float(np.mean(value)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "max": float(np.max(value)),
    }


def as_links(value: np.ndarray) -> np.ndarray:
    return value.reshape(len(value), -1, value.shape[-1])


def build_new_calibrated_oof(
    truth: np.ndarray,
    regimes: np.ndarray,
    folds: list[np.ndarray],
) -> np.memmap:
    if NEW_CALIBRATED_OOF.is_file() and NEW_CALIBRATED_DONE.is_file():
        cached = np.load(NEW_CALIBRATED_OOF, mmap_mode="r")
        if cached.shape == truth.shape:
            return cached
    NEW_CALIBRATED_OOF.unlink(missing_ok=True)
    NEW_CALIBRATED_DONE.unlink(missing_ok=True)
    mean = as_links(
        np.load(
            calibration.checkpoint_path(CHECKPOINT, "train", "mean"),
            mmap_mode="r",
        )
    )
    output = np.lib.format.open_memmap(
        NEW_CALIBRATED_OOF, mode="w+", dtype=np.float32, shape=truth.shape
    )
    all_ids = np.arange(len(truth), dtype=np.int64)
    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, query_ids)
        for start in range(0, truth.shape[1], LINK_BATCH):
            end = min(start + LINK_BATCH, truth.shape[1])
            ratios = v447.fit_ratio_chunk(truth, mean, fit_ids, regimes, start, end)
            output[query_ids, start:end] = v447.apply_ratio_chunk(
                mean, query_ids, regimes, ratios, start, end
            )
        output.flush()
        print(f"v554 conditional new-source fold {fold + 1}/5", flush=True)
    NEW_CALIBRATED_DONE.write_text("complete\n", encoding="ascii")
    return np.load(NEW_CALIBRATED_OOF, mmap_mode="r")


def balance_matrix(
    kernel: np.ndarray,
    row_target: np.ndarray,
    column_target: np.ndarray,
) -> np.ndarray:
    row_target = np.maximum(np.asarray(row_target, dtype=np.float64), 0.0)
    column_target = np.maximum(np.asarray(column_target, dtype=np.float64), 0.0)
    total = float(np.sum(row_target))
    if total <= TINY:
        return np.zeros_like(kernel, dtype=np.float32)
    column_target *= total / max(float(np.sum(column_target)), TINY)
    active_rows = row_target > TINY
    active_columns = column_target > TINY
    value = np.maximum(np.asarray(kernel, dtype=np.float64), FLOOR)
    value[~active_rows] = 0.0
    value[:, ~active_columns] = 0.0
    for _ in range(SINKHORN_ITERATIONS):
        row_sum = np.sum(value, axis=1)
        value[active_rows] *= (
            row_target[active_rows] / np.maximum(row_sum[active_rows], TINY)
        )[:, None]
        column_sum = np.sum(value, axis=0)
        value[:, active_columns] *= (
            column_target[active_columns] / np.maximum(column_sum[active_columns], TINY)
        )[None, :]
    row_sum = np.sum(value, axis=1)
    value[active_rows] *= (
        row_target[active_rows] / np.maximum(row_sum[active_rows], TINY)
    )[:, None]
    return value.astype(np.float32)


def conditional_shape(
    baseline_shape: np.ndarray,
    target_shape: np.ndarray,
    row_energy: np.ndarray | None = None,
) -> np.ndarray:
    links = baseline_shape.shape[0]
    if row_energy is None:
        row_target = np.ones(links, dtype=np.float64)
    else:
        row_target = np.asarray(row_energy, dtype=np.float64)
    baseline_mass = baseline_shape.astype(np.float64) * row_target[:, None]
    target_mass = target_shape.astype(np.float64) * row_target[:, None]
    balanced = balance_matrix(target_mass, row_target, baseline_mass.sum(axis=0))
    output = balanced / np.maximum(row_target[:, None], TINY)
    output[row_target <= TINY] = baseline_shape[row_target <= TINY]
    return v447.normalize_links(output)


def evaluate_oof(
    truth: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    regimes: np.ndarray,
    fold_id: np.ndarray,
) -> dict[str, object]:
    baseline_score = np.zeros(len(truth), dtype=np.float64)
    full_score = np.zeros(len(truth), dtype=np.float64)
    conditional_score = np.zeros(len(truth), dtype=np.float64)
    column_error = np.zeros(len(truth), dtype=np.float64)
    output_score = np.lib.format.open_memmap(
        OUT_DIR / "conditional_score_scratch.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(truth),),
    )
    for index in range(len(truth)):
        truth_point = np.asarray(truth[index], dtype=np.float32)
        old_point = np.asarray(old[index], dtype=np.float32)
        if regimes[index] in ("medium", "heavy"):
            new_point = np.asarray(new[index], dtype=np.float32)
            candidate = conditional_shape(old_point, new_point)
            full = new_point
        else:
            candidate = old_point
            full = old_point
        baseline_score[index] = float(
            np.mean(v447.link_cosine(truth_point[None], old_point[None]))
        )
        full_score[index] = float(
            np.mean(v447.link_cosine(truth_point[None], full[None]))
        )
        conditional_score[index] = float(
            np.mean(v447.link_cosine(truth_point[None], candidate[None]))
        )
        column_error[index] = float(
            np.max(np.abs(candidate.sum(axis=0) - old_point.sum(axis=0)))
        )
        output_score[index] = conditional_score[index]
        if (index + 1) % 200 == 0 or index + 1 == len(truth):
            output_score.flush()
            print(f"v554 conditional OOF {index + 1}/{len(truth)}", flush=True)
    (OUT_DIR / "conditional_score_scratch.npy").unlink(missing_ok=True)
    folds = []
    for fold in range(5):
        mask = fold_id == fold
        folds.append(
            {
                "fold": fold,
                "count": int(np.sum(mask)),
                "gain": float(np.mean(conditional_score[mask] - baseline_score[mask])),
            }
        )
    by_regime = {}
    for regime in calibration.REGIMES:
        mask = regimes == regime
        by_regime[regime] = {
            "count": int(np.sum(mask)),
            "baseline": float(np.mean(baseline_score[mask])),
            "full": float(np.mean(full_score[mask])),
            "conditional": float(np.mean(conditional_score[mask])),
            "gain": float(np.mean(conditional_score[mask] - baseline_score[mask])),
        }
    np.savez_compressed(
        CONDITIONAL_OOF,
        fold_id=fold_id,
        regimes=regimes,
        baseline_score=baseline_score,
        full_score=full_score,
        conditional_score=conditional_score,
        column_error=column_error,
    )
    return {
        "baseline": summarize(baseline_score),
        "full": summarize(full_score),
        "conditional": summarize(conditional_score),
        "full_gain": float(np.mean(full_score - baseline_score)),
        "conditional_gain": float(np.mean(conditional_score - baseline_score)),
        "retained_fraction": float(
            np.mean(conditional_score - baseline_score)
            / max(np.mean(full_score - baseline_score), TINY)
        ),
        "fold_wins": int(sum(row["gain"] > 0.0 for row in folds)),
        "folds": folds,
        "by_regime": by_regime,
        "column_error": summarize(column_error),
    }


def generate_test_probe(test_regimes: np.ndarray) -> dict[str, object]:
    base = np.load(BASE, mmap_mode="r")
    action = np.load(FULL_ACTION, mmap_mode="r")
    desired_out = np.lib.format.open_memmap(
        TEST_DESIRED,
        mode="w+",
        dtype=np.float32,
        shape=(len(base), base.shape[1] * base.shape[2], base.shape[3]),
    )
    output = np.lib.format.open_memmap(
        TEST_OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    diagnostics = {
        "pdp_l1_per_link": np.zeros(len(base), dtype=np.float32),
        "column_error": np.zeros(len(base), dtype=np.float32),
        "sh_l1": np.zeros(len(base), dtype=np.float32),
        "pdp_target_error": np.zeros(len(base), dtype=np.float32),
        "power_ratio": np.ones(len(base), dtype=np.float32),
    }
    for start in range(0, len(base), BATCH):
        end = min(start + BATCH, len(base))
        base_h = np.asarray(base[start:end], dtype=np.complex64)
        base_pdp, link_energy = v426.linkwise_pdp(base_h)
        local_action = np.asarray(action[start:end], dtype=np.float32).reshape(
            end - start, -1, base.shape[-1]
        )
        raw_target = v447.normalize_links(
            base_pdp.reshape(end - start, -1, base.shape[-1])
            * np.exp(local_action).astype(np.float32)
        )
        desired = np.empty_like(raw_target)
        for row in range(end - start):
            index = start + row
            base_shape = base_pdp[row].reshape(-1, base.shape[-1])
            if test_regimes[index] in ("medium", "heavy"):
                desired[row] = conditional_shape(
                    base_shape,
                    raw_target[row],
                    link_energy[row].reshape(-1),
                )
            else:
                desired[row] = base_shape
        desired_out[start:end] = desired
        desired_4d = desired.reshape(base_pdp.shape)
        base_support, _ = v423.leaked_support(base_h)
        base_pas = v423.leaked_marginals(base_support)
        candidate = base_h.copy()
        for _ in range(PROJECTION_CYCLES):
            candidate = v423.project_pas(candidate, base_pas)
            candidate = v426.restore_desired_pdp(candidate, base_h, desired_4d)
        output[start:end] = candidate
        actual_pdp, _ = v426.linkwise_pdp(candidate)
        actual_support, _ = v423.leaked_support(candidate)
        actual_pas = v423.leaked_marginals(actual_support)
        diagnostics["pdp_l1_per_link"][start:end] = np.sum(
            np.abs(actual_pdp - base_pdp),
            axis=(1, 2, 3),
            dtype=np.float64,
        ) / np.float32(base.shape[1] * base.shape[2])
        base_mass = base_pdp * link_energy[..., None]
        desired_mass = desired_4d * link_energy[..., None]
        diagnostics["column_error"][start:end] = np.max(
            np.abs(desired_mass.sum(axis=(1, 2)) - base_mass.sum(axis=(1, 2))),
            axis=1,
        ).astype(np.float32)
        diagnostics["sh_l1"][start:end] = np.sum(
            np.abs(actual_pas["sh"] - base_pas["sh"]),
            axis=(1, 2),
            dtype=np.float64,
        ).astype(np.float32)
        diagnostics["pdp_target_error"][start:end] = np.max(
            np.abs(actual_pdp - desired_4d), axis=(1, 2, 3)
        ).astype(np.float32)
        base_power = np.sum(np.abs(base_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
        candidate_power = np.sum(
            np.abs(candidate) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        diagnostics["power_ratio"][start:end] = (
            candidate_power / np.maximum(base_power, TINY)
        ).astype(np.float32)
        if end == len(base) or end % 20 == 0:
            desired_out.flush()
            output.flush()
            print(f"v554 conditional test {end}/{len(base)}", flush=True)
    base_replay = replay.score_prediction(BASE)
    probe_replay = replay.score_prediction(TEST_OUTPUT)
    return {
        "diagnostics": {name: summarize(value) for name, value in diagnostics.items()},
        "replay": {
            "base": base_replay,
            "probe": probe_replay,
            "traceback_delta": float(
                probe_replay["score_traceback_leaked"]
                - base_replay["score_traceback_leaked"]
            ),
            "pdp_delta": float(probe_replay["pdp"] - base_replay["pdp"]),
            "leaked_pas_delta": float(
                probe_replay["leaked_sh"] - base_replay["leaked_sh"]
            ),
        },
    }


def main() -> None:
    for path in (TRUTH, OLD_OOF, BASE, FULL_ACTION):
        require(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    truth = np.load(TRUTH, mmap_mode="r")
    old = np.load(OLD_OOF, mmap_mode="r")
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geometry["train_blocked"].astype(np.float32)).astype("U8")
    test_regimes = regime_of(geometry["test_blocked"].astype(np.float32)).astype("U8")
    all_ids = np.arange(len(truth), dtype=np.int64)
    folds = v401.confirmatory_folds(all_ids, positions[:, :2])
    fold_id = np.full(len(truth), -1, dtype=np.int16)
    for fold, ids in enumerate(folds):
        fold_id[ids] = fold
    new = build_new_calibrated_oof(truth, regimes, folds)
    oof = evaluate_oof(truth, old, new, regimes, fold_id)
    test = generate_test_probe(test_regimes)
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The v554 multi-seed source should correct conditional link-delay allocation "
            "while fixed global delay mass removes its conflict with v549 PAS-S."
        ),
        "method_definition": {
            "kernel": "calibrated 24-seed linkwise-PDP target",
            "row_constraint": "preserve every baseline link energy",
            "column_constraint": "preserve each point's baseline global 192-bin delay mass",
            "solver": f"{SINKHORN_ITERATIONS} iterations of matrix balancing",
            "regimes": ["medium", "heavy"],
            "projection_cycles": PROJECTION_CYCLES,
        },
        "oof": oof,
        "test_probe": test,
        "deployment_gate": {
            "pass": bool(
                oof["conditional_gain"] > 0.0
                and oof["fold_wins"] >= 4
                and test["replay"]["traceback_delta"] > -0.02
            ),
            "requirements": "positive OOF, at least 4/5 folds, replay structural loss above -0.02",
        },
        "outputs": {
            "new_calibrated_oof": PATHS.relative(NEW_CALIBRATED_OOF),
            "oof": PATHS.relative(CONDITIONAL_OOF),
            "test_desired": PATHS.relative(TEST_DESIRED),
            "test_probe": PATHS.relative(TEST_OUTPUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
