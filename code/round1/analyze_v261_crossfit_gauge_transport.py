#!/usr/bin/env python3
"""v261: strict cross-fit propagation-gauge spectral transport."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v255_global_spectral_transport_oracle as v255
import analyze_v260_shift_graph_synchronization_oracle as v260
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 261
VERSION = "v261_crossfit_gauge_transport"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v261_Round1_Test_Channel.npy"
TARGETS = OUT_DIR / "v261_gauge_transport_targets.npz"
ACTIONS = OUT_DIR / "v261_gauge_transport_actions.npz"
GAUGES = OUT_DIR / "v261_train_test_gauges.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"

LOCAL_GATE = 0.645604
FOLD_SEEDS = v254.FOLD_SEEDS
VAL_COUNT = 320
SPECTRAL_NEIGHBORS = 16
GRAPH_NEIGHBORS = 16
POWER = 2.0
SHIFT_BATCH = 128
D_BOUND = 4
S_BOUND = 4
H_BOUND = 3
GAUGE_K = (4, 8, 16, 24, 32)
GAUGE_POWER = (1.0, 2.0, 3.0)
MODES = ("d", "sh", "joint")
BLENDS = (0.25, 0.5, 0.75, 1.0)
MIN_OFFICIAL_DELTA = 5e-4


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


def compute_subset_gauges(
    subset_ids: np.ndarray,
    pos: np.ndarray,
    sh: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    subset_pos = pos[subset_ids]
    distance, neighbour_local = cKDTree(subset_pos).query(
        subset_pos, k=GRAPH_NEIGHBORS + 1
    )
    distance = distance[:, 1:]
    neighbour_local = neighbour_local[:, 1:]
    weight = 1.0 / np.maximum(distance, 0.05) ** POWER
    weight /= weight.sum(axis=1, keepdims=True)
    d_shift = np.empty((len(subset_ids), GRAPH_NEIGHBORS), dtype=np.int16)
    s_shift = np.empty_like(d_shift)
    h_shift = np.empty_like(d_shift)
    subset_sh = sh[subset_ids]
    subset_d = d[subset_ids]
    for start in range(0, len(subset_ids), SHIFT_BATCH):
        end = min(start + SHIFT_BATCH, len(subset_ids))
        neighbour_sh = subset_sh[neighbour_local[start:end]]
        neighbour_d = subset_d[neighbour_local[start:end]]
        d_shift[start:end] = v255.best_1d_shifts(
            subset_d[start:end], neighbour_d, D_BOUND
        )
        s_shift[start:end], h_shift[start:end] = v255.best_2d_shifts(
            subset_sh[start:end], neighbour_sh, S_BOUND, H_BOUND
        )
    source = np.repeat(np.arange(len(subset_ids), dtype=np.int64), GRAPH_NEIGHBORS)
    target = neighbour_local.reshape(-1).astype(np.int64)
    edge_weight = weight.reshape(-1).astype(np.float64)
    gauge_columns: list[np.ndarray] = []
    solver: dict[str, object] = {}
    for name, shift in (("d", d_shift), ("s", s_shift), ("h", h_shift)):
        gauge, solve_report = v260.solve_gauge(
            source,
            target,
            shift.reshape(-1),
            edge_weight,
            len(subset_ids),
        )
        gauge_columns.append(gauge)
        solver[name] = solve_report
    gauge = np.stack(gauge_columns, axis=1).astype(np.float32)
    return gauge, {
        "node_count": int(len(subset_ids)),
        "edge_count": int(len(source)),
        "solver": solver,
        "gauge_d": summarize(gauge[:, 0]),
        "gauge_s": summarize(gauge[:, 1]),
        "gauge_h": summarize(gauge[:, 2]),
    }


def spectral_bank(
    fit_ids: np.ndarray,
    query_pos: np.ndarray,
    pos: np.ndarray,
    sh: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance, local = cKDTree(pos[fit_ids]).query(
        query_pos, k=SPECTRAL_NEIGHBORS
    )
    weight = 1.0 / np.maximum(distance, 0.05) ** POWER
    weight /= weight.sum(axis=1, keepdims=True)
    absolute = fit_ids[local]
    neighbour_sh = sh[absolute]
    neighbour_d = d[absolute]
    return (
        local.astype(np.int64),
        weight.astype(np.float32),
        neighbour_sh,
        neighbour_d,
        weighted_average(neighbour_sh, weight),
        weighted_average(neighbour_d, weight),
    )


def interpolate_gauge(
    fit_pos: np.ndarray,
    fit_gauge: np.ndarray,
    query_pos: np.ndarray,
    k: int,
    power: float,
) -> np.ndarray:
    distance, local = cKDTree(fit_pos).query(query_pos, k=k)
    if k == 1:
        distance = distance[:, None]
        local = local[:, None]
    weight = 1.0 / np.maximum(distance, 0.05) ** power
    weight /= weight.sum(axis=1, keepdims=True)
    return np.sum(weight[:, :, None] * fit_gauge[local], axis=1).astype(np.float32)


def aligned_targets(
    query_gauge: np.ndarray,
    fit_gauge: np.ndarray,
    neighbour_local: np.ndarray,
    weight: np.ndarray,
    neighbour_sh: np.ndarray,
    neighbour_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    difference = query_gauge[:, None, :] - fit_gauge[neighbour_local]
    d_shift = np.clip(np.rint(difference[:, :, 0]), -D_BOUND, D_BOUND).astype(np.int16)
    s_shift = np.clip(np.rint(difference[:, :, 1]), -S_BOUND, S_BOUND).astype(np.int16)
    h_shift = np.clip(np.rint(difference[:, :, 2]), -H_BOUND, H_BOUND).astype(np.int16)
    aligned_sh = weighted_average(
        v255.apply_2d_shifts(neighbour_sh, s_shift, h_shift), weight
    )
    aligned_d = weighted_average(
        v255.apply_1d_shifts(neighbour_d, d_shift), weight
    )
    return aligned_sh, aligned_d, {
        "d_shift": d_shift,
        "s_shift": s_shift,
        "h_shift": h_shift,
    }


def blend_targets(
    raw_sh: np.ndarray,
    raw_d: np.ndarray,
    aligned_sh: np.ndarray,
    aligned_d: np.ndarray,
    mode: str,
    blend: float,
) -> tuple[np.ndarray, np.ndarray]:
    out_sh = raw_sh
    out_d = raw_d
    if mode in {"sh", "joint"}:
        out_sh = normalize((1.0 - blend) * raw_sh + blend * aligned_sh)
    if mode in {"d", "joint"}:
        out_d = normalize((1.0 - blend) * raw_d + blend * aligned_d)
    return out_sh, out_d


def main() -> None:
    for path in (TARGET_CACHE, BASE):
        if not path.is_file():
            raise FileNotFoundError(path)
    pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)[:, :2]
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)[:, :2]
    data = np.load(TARGET_CACHE)
    sh = data["sh"].astype(np.float32)
    d = data["d"].astype(np.float32)
    valid_ids = np.flatnonzero(data["valid"].astype(bool)).astype(np.int64)

    config_names = [
        f"k{k}_p{power:g}_{mode}_b{blend:g}"
        for k in GAUGE_K
        for power in GAUGE_POWER
        for mode in MODES
        for blend in BLENDS
    ]
    config_results: dict[str, list[dict[str, float]]] = {
        name: [] for name in config_names
    }
    raw_results: list[dict[str, float]] = []
    fold_reports: list[dict[str, object]] = []
    for fold_index, seed in enumerate(FOLD_SEEDS):
        matched = v254.choose_seeded_matched_indices(pos, test_pos, VAL_COUNT, seed)
        val_ids = np.intersect1d(matched, valid_ids)
        fit_ids = np.setdiff1d(valid_ids, val_ids)
        fit_gauge, gauge_report = compute_subset_gauges(fit_ids, pos, sh, d)
        (
            neighbour_local,
            weight,
            neighbour_sh,
            neighbour_d,
            raw_sh,
            raw_d,
        ) = spectral_bank(fit_ids, pos[val_ids], pos, sh, d)
        raw_results.append(metrics(raw_sh, raw_d, sh[val_ids], d[val_ids]))
        for k in GAUGE_K:
            for power in GAUGE_POWER:
                query_gauge = interpolate_gauge(
                    pos[fit_ids], fit_gauge, pos[val_ids], k, power
                )
                aligned_sh, aligned_d, _ = aligned_targets(
                    query_gauge,
                    fit_gauge,
                    neighbour_local,
                    weight,
                    neighbour_sh,
                    neighbour_d,
                )
                for mode in MODES:
                    for blend in BLENDS:
                        name = f"k{k}_p{power:g}_{mode}_b{blend:g}"
                        pred_sh, pred_d = blend_targets(
                            raw_sh, raw_d, aligned_sh, aligned_d, mode, blend
                        )
                        config_results[name].append(
                            metrics(pred_sh, pred_d, sh[val_ids], d[val_ids])
                        )
        fold_reports.append(
            {
                "fold": fold_index,
                "seed": seed,
                "fit_count": int(len(fit_ids)),
                "validation_count": int(len(val_ids)),
                "gauge": gauge_report,
            }
        )
        print(f"v261 official fold {fold_index + 1}/{len(FOLD_SEEDS)}", flush=True)

    raw_mean = {
        key: float(np.mean([row[key] for row in raw_results]))
        for key in ("sh", "d", "weighted")
    }
    summary: dict[str, dict[str, object]] = {}
    for name, rows in config_results.items():
        weighted = np.asarray([row["weighted"] for row in rows])
        summary[name] = {
            "mean": {
                key: float(np.mean([row[key] for row in rows]))
                for key in ("sh", "d", "weighted")
            },
            "std_weighted": float(np.std(weighted)),
            "folds": rows,
        }
    best_name = max(summary, key=lambda name: float(summary[name]["mean"]["weighted"]))
    raw_fold = np.asarray([row["weighted"] for row in raw_results])
    best_fold = np.asarray([row["weighted"] for row in config_results[best_name]])
    official_delta = float(best_fold.mean() - raw_fold.mean())
    official_wins = int(np.sum(best_fold > raw_fold))
    official_selected = bool(official_delta >= MIN_OFFICIAL_DELTA and official_wins >= 3)

    local_score = None
    actions = None
    final_gauge_report = None
    shift_diagnostics = None
    if official_selected:
        parts = best_name.split("_")
        k = int(parts[0][1:])
        power = float(parts[1][1:])
        mode = parts[2]
        blend = float(parts[3][1:])
        fit_gauge, final_gauge_report = compute_subset_gauges(valid_ids, pos, sh, d)
        (
            neighbour_local,
            weight,
            neighbour_sh,
            neighbour_d,
            raw_sh,
            raw_d,
        ) = spectral_bank(valid_ids, test_pos, pos, sh, d)
        test_gauge = interpolate_gauge(pos[valid_ids], fit_gauge, test_pos, k, power)
        aligned_sh, aligned_d, shifts = aligned_targets(
            test_gauge,
            fit_gauge,
            neighbour_local,
            weight,
            neighbour_sh,
            neighbour_d,
        )
        transported_sh, transported_d = blend_targets(
            raw_sh, raw_d, aligned_sh, aligned_d, mode, blend
        )
        base = np.load(BASE, mmap_mode="r")
        current_sh, current_d = v247.extract_current(np.asarray(base, dtype=np.complex64))
        target_sh = normalize(current_sh + transported_sh - raw_sh)
        target_d = normalize(current_d + transported_d - raw_d)
        np.savez_compressed(
            TARGETS,
            sh=target_sh,
            d=target_d,
            raw_sh=raw_sh,
            raw_d=raw_d,
            aligned_sh=aligned_sh,
            aligned_d=aligned_d,
            transported_sh=transported_sh,
            transported_d=transported_d,
            method=np.asarray(best_name),
        )
        np.savez_compressed(
            GAUGES,
            valid_ids=valid_ids,
            train_gauge=fit_gauge,
            test_gauge=test_gauge,
            **shifts,
        )
        actions = v247.project(target_sh, target_d, OUTPUT)
        np.savez_compressed(ACTIONS, **actions)
        local_score = v208.score_prediction(OUTPUT)
        shift_diagnostics = {
            key: summarize(value.reshape(-1)) for key, value in shifts.items()
        }

    ranking = sorted(
        summary,
        key=lambda name: float(summary[name]["mean"]["weighted"]),
        reverse=True,
    )
    report = {
        "method": VERSION,
        "status": (
            "mechanism_gate_passed_needs_neural_gauge_predictor"
            if local_score is not None
            and float(local_score["score_traceback_leaked"]) > LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "Fit-only synchronized propagation gauges can be interpolated to unseen official "
            "coordinates and recover a substantial fraction of the v260 alignment gain."
        ),
        "compliance_boundary": {
            "uses": [
                "official fit-only train H pair shifts",
                "official train/test coordinates",
                "strict v239 carrier for residual deployment",
            ],
            "does_not_use": [
                "validation H while solving fit gauges",
                "synthetic or Sionna labels",
                "teacher output",
                "local replay truth for selection",
                "hidden labels",
                "raw RT H",
            ],
            "candidate_note": (
                "This KNN gauge interpolator is mechanism analysis; if it passes the gate, "
                "a learned Physical-AI gauge predictor is required before packaging."
            ),
        },
        "parameters": {
            "fold_seeds": list(FOLD_SEEDS),
            "graph_neighbours": GRAPH_NEIGHBORS,
            "spectral_neighbours": SPECTRAL_NEIGHBORS,
            "gauge_k": list(GAUGE_K),
            "gauge_power": list(GAUGE_POWER),
            "modes": list(MODES),
            "blends": list(BLENDS),
        },
        "folds": fold_reports,
        "selection": {
            "raw_mean": raw_mean,
            "best_name": best_name,
            "best": summary[best_name],
            "official_delta": official_delta,
            "official_wins": official_wins,
            "official_selected": official_selected,
            "top_20": [{"name": name, **summary[name]} for name in ranking[:20]],
        },
        "final_gauge": final_gauge_report,
        "test_shift_diagnostics": shift_diagnostics,
        "projection_diagnostics": (
            {key: summarize(value) for key, value in actions.items()}
            if actions is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(float(local_score["score_traceback_leaked"]) > LOCAL_GATE),
                "gap": float(local_score["score_traceback_leaked"]) - LOCAL_GATE,
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "targets": PATHS.relative(TARGETS) if TARGETS.is_file() else None,
            "actions": PATHS.relative(ACTIONS) if ACTIONS.is_file() else None,
            "gauges": PATHS.relative(GAUGES) if GAUGES.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
