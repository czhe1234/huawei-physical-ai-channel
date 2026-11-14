#!/usr/bin/env python3
"""v256: deployable multi-reference spectral self-alignment audit.

Unlike the v255 oracle, this experiment never uses a query truth spectrum to
estimate a shift. It initializes a reference from the query's labelled
neighbours, aligns each neighbour to that reference by restricted circular
cross-correlation, and re-averages. Hyperparameters are selected only on five
official matched holdouts. Local replay truth is read once after a stable
official winner has been fixed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v255_global_spectral_transport_oracle as v255
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 256
VERSION = "v256_self_aligned_spectral_transport"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v256_Round1_Test_Channel.npy"
TARGETS = OUT_DIR / "v256_self_aligned_spectral_targets.npz"
ACTIONS = OUT_DIR / "v256_self_aligned_spectral_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

LOCAL_GATE = 0.645604
FOLD_SEEDS = v254.FOLD_SEEDS
VAL_COUNT = 320
NEIGHBORS = 16
POWER = 2.0
MIN_OFFICIAL_DELTA = 5e-4
BOUNDS = ((2, 2, 2), (4, 4, 3), (8, 8, 4))
INITIALIZATIONS = ("raw", "nearest")
ITERATIONS = (1, 2, 3)
BLENDS = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Config:
    initialization: str
    d_bound: int
    s_bound: int
    h_bound: int
    iterations: int
    blend: float

    @property
    def name(self) -> str:
        return (
            f"{self.initialization}_d{self.d_bound}_s{self.s_bound}_h{self.h_bound}"
            f"_iter{self.iterations}_blend{self.blend:g}"
        )


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    total = out.reshape(len(out), -1).sum(axis=1)
    valid = total > 1e-20
    out[valid] /= total[valid].reshape(
        (int(np.sum(valid)),) + (1,) * (out.ndim - 1)
    )
    out[~valid] = 0.0
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def metrics(
    pred_sh: np.ndarray,
    pred_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> dict[str, float]:
    sh = float(np.mean(row_cos(pred_sh, truth_sh)))
    d = float(np.mean(row_cos(pred_d, truth_d)))
    return {"sh": sh, "d": d, "weighted": 0.52 * sh + 0.48 * d}


def weighted_average(bank: np.ndarray, weight: np.ndarray) -> np.ndarray:
    shape = weight.shape + (1,) * (bank.ndim - 2)
    return normalize(
        np.sum(bank * weight.reshape(shape), axis=1, dtype=np.float64).astype(np.float32)
    )


def neighbour_bank(
    bank_pos: np.ndarray,
    query_pos: np.ndarray,
    bank_ids: np.ndarray,
    sh: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance, local = cKDTree(bank_pos).query(query_pos, k=NEIGHBORS)
    absolute = bank_ids[local]
    weight = 1.0 / np.maximum(distance, 0.05) ** POWER
    weight /= weight.sum(axis=1, keepdims=True)
    neighbour_sh = sh[absolute]
    neighbour_d = d[absolute]
    raw_sh = weighted_average(neighbour_sh, weight)
    raw_d = weighted_average(neighbour_d, weight)
    return neighbour_sh, neighbour_d, weight, raw_sh, raw_d


def alignment_trajectory(
    neighbour_sh: np.ndarray,
    neighbour_d: np.ndarray,
    weight: np.ndarray,
    initialization: str,
    d_bound: int,
    s_bound: int,
    h_bound: int,
    max_iterations: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    raw_sh = weighted_average(neighbour_sh, weight)
    raw_d = weighted_average(neighbour_d, weight)
    if initialization == "raw":
        current_sh = raw_sh
        current_d = raw_d
    elif initialization == "nearest":
        current_sh = normalize(neighbour_sh[:, 0])
        current_d = normalize(neighbour_d[:, 0])
    else:
        raise ValueError(initialization)

    trajectory: list[tuple[np.ndarray, np.ndarray]] = []
    last_d_shift = np.zeros(weight.shape, dtype=np.int16)
    last_s_shift = np.zeros(weight.shape, dtype=np.int16)
    last_h_shift = np.zeros(weight.shape, dtype=np.int16)
    for _ in range(max_iterations):
        last_d_shift = v255.best_1d_shifts(current_d, neighbour_d, d_bound)
        last_s_shift, last_h_shift = v255.best_2d_shifts(
            current_sh, neighbour_sh, s_bound, h_bound
        )
        current_d = weighted_average(
            v255.apply_1d_shifts(neighbour_d, last_d_shift), weight
        )
        current_sh = weighted_average(
            v255.apply_2d_shifts(neighbour_sh, last_s_shift, last_h_shift), weight
        )
        trajectory.append((current_sh, current_d))
    return trajectory, {
        "d_shift": last_d_shift,
        "s_shift": last_s_shift,
        "h_shift": last_h_shift,
    }


def main() -> None:
    if not TARGET_CACHE.is_file():
        raise FileNotFoundError(TARGET_CACHE)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)[:, :2]
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)[:, :2]
    data = np.load(TARGET_CACHE)
    sh = data["sh"].astype(np.float32)
    d = data["d"].astype(np.float32)
    valid_ids = np.flatnonzero(data["valid"].astype(bool)).astype(np.int64)

    configs = [
        Config(initialization, d_bound, s_bound, h_bound, iterations, blend)
        for initialization in INITIALIZATIONS
        for d_bound, s_bound, h_bound in BOUNDS
        for iterations in ITERATIONS
        for blend in BLENDS
    ]
    results: dict[str, list[dict[str, float]]] = {config.name: [] for config in configs}
    raw_results: list[dict[str, float]] = []
    fold_meta: list[dict[str, int]] = []
    for fold_index, seed in enumerate(FOLD_SEEDS):
        matched = v254.choose_seeded_matched_indices(train_pos, test_pos, VAL_COUNT, seed)
        val_ids = np.intersect1d(matched, valid_ids)
        fit_ids = np.setdiff1d(valid_ids, val_ids)
        neighbour_sh, neighbour_d, weight, raw_sh, raw_d = neighbour_bank(
            train_pos[fit_ids], train_pos[val_ids], fit_ids, sh, d
        )
        truth_sh = sh[val_ids]
        truth_d = d[val_ids]
        raw_results.append(metrics(raw_sh, raw_d, truth_sh, truth_d))
        fold_meta.append(
            {
                "fold": fold_index,
                "seed": seed,
                "fit_count": int(len(fit_ids)),
                "validation_count": int(len(val_ids)),
            }
        )
        for initialization in INITIALIZATIONS:
            for d_bound, s_bound, h_bound in BOUNDS:
                trajectory, _ = alignment_trajectory(
                    neighbour_sh,
                    neighbour_d,
                    weight,
                    initialization,
                    d_bound,
                    s_bound,
                    h_bound,
                    max(ITERATIONS),
                )
                for iterations in ITERATIONS:
                    aligned_sh, aligned_d = trajectory[iterations - 1]
                    for blend in BLENDS:
                        pred_sh = normalize((1.0 - blend) * raw_sh + blend * aligned_sh)
                        pred_d = normalize((1.0 - blend) * raw_d + blend * aligned_d)
                        config = Config(
                            initialization,
                            d_bound,
                            s_bound,
                            h_bound,
                            iterations,
                            blend,
                        )
                        results[config.name].append(
                            metrics(pred_sh, pred_d, truth_sh, truth_d)
                        )
        print(f"v256 official fold {fold_index + 1}/{len(FOLD_SEEDS)}", flush=True)

    raw_mean = {
        key: float(np.mean([row[key] for row in raw_results]))
        for key in ("sh", "d", "weighted")
    }
    summary: dict[str, dict[str, object]] = {}
    for config in configs:
        rows = results[config.name]
        weighted = np.asarray([row["weighted"] for row in rows], dtype=np.float64)
        summary[config.name] = {
            "config": {
                "initialization": config.initialization,
                "d_bound": config.d_bound,
                "s_bound": config.s_bound,
                "h_bound": config.h_bound,
                "iterations": config.iterations,
                "blend": config.blend,
            },
            "mean": {
                key: float(np.mean([row[key] for row in rows]))
                for key in ("sh", "d", "weighted")
            },
            "std_weighted": float(np.std(weighted)),
            "folds": rows,
        }
    best_name = max(summary, key=lambda name: float(summary[name]["mean"]["weighted"]))
    best = summary[best_name]
    raw_fold = np.asarray([row["weighted"] for row in raw_results])
    best_fold = np.asarray([row["weighted"] for row in results[best_name]])
    official_delta = float(best_fold.mean() - raw_fold.mean())
    official_wins = int(np.sum(best_fold > raw_fold))
    official_selected = bool(official_delta >= MIN_OFFICIAL_DELTA and official_wins >= 3)

    local_score = None
    actions = None
    test_shift = None
    if official_selected:
        neighbour_sh, neighbour_d, weight, raw_sh, raw_d = neighbour_bank(
            train_pos[valid_ids], test_pos, valid_ids, sh, d
        )
        best_config = best["config"]
        trajectory, test_shift = alignment_trajectory(
            neighbour_sh,
            neighbour_d,
            weight,
            str(best_config["initialization"]),
            int(best_config["d_bound"]),
            int(best_config["s_bound"]),
            int(best_config["h_bound"]),
            int(best_config["iterations"]),
        )
        aligned_sh, aligned_d = trajectory[int(best_config["iterations"]) - 1]
        blend = float(best_config["blend"])
        test_sh = normalize((1.0 - blend) * raw_sh + blend * aligned_sh)
        test_d = normalize((1.0 - blend) * raw_d + blend * aligned_d)
        np.savez_compressed(
            TARGETS,
            sh=test_sh,
            d=test_d,
            raw_sh=raw_sh,
            raw_d=raw_d,
            method=np.asarray(best_name),
            **test_shift,
        )
        actions = v247.project(test_sh, test_d, OUTPUT)
        np.savez_compressed(ACTIONS, **actions)
        local_score = v208.score_prediction(OUTPUT)

    ranking = sorted(
        summary,
        key=lambda name: float(summary[name]["mean"]["weighted"]),
        reverse=True,
    )
    report = {
        "method": VERSION,
        "status": (
            "mechanism_gate_passed_needs_neural_compliance_layer"
            if local_score is not None
            and float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "The v255 oracle gain can be recovered without query truth by iteratively aligning "
            "each labelled neighbour to a multi-reference consensus spectrum."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H leaked S-H and PDP targets",
                "official train/test coordinates",
                "strict v239 official-data carrier after official selection",
            ],
            "does_not_use": [
                "validation truth to estimate deployed shifts",
                "synthetic or Sionna channel labels",
                "teacher outputs",
                "local replay truth for method selection",
                "hidden labels",
                "raw RT H",
            ],
            "candidate_note": (
                "This deterministic mechanism is analysis-only; a final package would add an "
                "official-train neural confidence gate so the submitted method remains Physical AI."
            ),
        },
        "parameters": {
            "fold_seeds": list(FOLD_SEEDS),
            "neighbours": NEIGHBORS,
            "power": POWER,
            "bounds": [list(item) for item in BOUNDS],
            "initializations": list(INITIALIZATIONS),
            "iterations": list(ITERATIONS),
            "blends": list(BLENDS),
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
        },
        "folds": fold_meta,
        "selection": {
            "raw_mean": raw_mean,
            "best_name": best_name,
            "best": best,
            "official_delta": official_delta,
            "official_wins": official_wins,
            "official_selected": official_selected,
            "top_20": [{"name": name, **summary[name]} for name in ranking[:20]],
        },
        "test_shift_diagnostics": (
            {key: summarize(value.reshape(-1)) for key, value in test_shift.items()}
            if test_shift is not None
            else None
        ),
        "projection_diagnostics": (
            {key: summarize(value) for key, value in actions.items()}
            if actions is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(
                    float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
                ),
                "gap": float(local_score["score_traceback_leaked"]) - LOCAL_GATE,
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "targets": PATHS.relative(TARGETS) if TARGETS.is_file() else None,
            "actions": PATHS.relative(ACTIONS) if ACTIONS.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
