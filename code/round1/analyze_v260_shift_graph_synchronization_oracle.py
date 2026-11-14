#!/usr/bin/env python3
"""v260: graph synchronization audit for official pairwise spectral shifts."""

from __future__ import annotations

import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

import analyze_v255_global_spectral_transport_oracle as v255
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 260
VERSION = "v260_shift_graph_synchronization_oracle"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

NEIGHBORS = 16
POWER = 2.0
D_BOUND = 4
S_BOUND = 4
H_BOUND = 3
SHIFT_BATCH = 128
ANCHOR_WEIGHT = 1e-2


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


def solve_gauge(
    source: np.ndarray,
    target: np.ndarray,
    shift: np.ndarray,
    weight: np.ndarray,
    node_count: int,
) -> tuple[np.ndarray, dict[str, float]]:
    edge_count = len(source)
    sqrt_weight = np.sqrt(np.maximum(weight, 1e-8))
    rows = np.repeat(np.arange(edge_count, dtype=np.int64), 2)
    cols = np.stack([source, target], axis=1).reshape(-1)
    values = np.stack([sqrt_weight, -sqrt_weight], axis=1).reshape(-1)
    incidence = sparse.coo_matrix(
        (values, (rows, cols)), shape=(edge_count, node_count)
    ).tocsr()
    anchor = sparse.csr_matrix(
        ([ANCHOR_WEIGHT], ([0], [0])), shape=(1, node_count)
    )
    matrix = sparse.vstack([incidence, anchor], format="csr")
    rhs = np.concatenate([sqrt_weight * shift.astype(np.float64), [0.0]])
    solution = lsqr(matrix, rhs, atol=1e-9, btol=1e-9, iter_lim=5000)
    gauge = solution[0].astype(np.float32)
    return gauge, {
        "iterations": int(solution[2]),
        "residual_norm": float(solution[3]),
        "condition_estimate": float(solution[6]),
    }


def edge_diagnostics(
    gauge: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    truth: np.ndarray,
    bound: int,
) -> dict[str, object]:
    continuous = gauge[source] - gauge[target]
    rounded = np.clip(np.rint(continuous), -bound, bound).astype(np.int16)
    return {
        "zero_accuracy": float(np.mean(truth == 0)),
        "gauge_exact_accuracy": float(np.mean(rounded == truth)),
        "gauge_within_one": float(np.mean(np.abs(rounded - truth) <= 1)),
        "continuous_mae": float(np.mean(np.abs(continuous - truth))),
        "rounded_mae": float(np.mean(np.abs(rounded - truth))),
        "gauge": summarize(gauge),
        "edge_prediction": summarize(continuous),
        "truth": summarize(truth),
    }


def main() -> None:
    if not TARGET_CACHE.is_file():
        raise FileNotFoundError(TARGET_CACHE)
    pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)[:, :2]
    data = np.load(TARGET_CACHE)
    sh_all = data["sh"].astype(np.float32)
    d_all = data["d"].astype(np.float32)
    valid_ids = np.flatnonzero(data["valid"].astype(bool)).astype(np.int64)
    pos_valid = pos[valid_ids]
    sh = sh_all[valid_ids]
    d = d_all[valid_ids]
    distance, local = cKDTree(pos_valid).query(pos_valid, k=NEIGHBORS + 1)
    distance = distance[:, 1:]
    neighbour_local = local[:, 1:]
    weight = 1.0 / np.maximum(distance, 0.05) ** POWER
    weight /= weight.sum(axis=1, keepdims=True)

    d_shift = np.empty((len(valid_ids), NEIGHBORS), dtype=np.int16)
    s_shift = np.empty_like(d_shift)
    h_shift = np.empty_like(d_shift)
    for start in range(0, len(valid_ids), SHIFT_BATCH):
        end = min(start + SHIFT_BATCH, len(valid_ids))
        neighbour_sh = sh[neighbour_local[start:end]]
        neighbour_d = d[neighbour_local[start:end]]
        d_shift[start:end] = v255.best_1d_shifts(
            d[start:end], neighbour_d, D_BOUND
        )
        s_shift[start:end], h_shift[start:end] = v255.best_2d_shifts(
            sh[start:end], neighbour_sh, S_BOUND, H_BOUND
        )
        print(f"v260 shift labels {end}/{len(valid_ids)}", flush=True)

    source = np.repeat(np.arange(len(valid_ids), dtype=np.int64), NEIGHBORS)
    target = neighbour_local.reshape(-1).astype(np.int64)
    edge_weight = weight.reshape(-1).astype(np.float64)
    gauges: dict[str, np.ndarray] = {}
    solver: dict[str, object] = {}
    diagnostics: dict[str, object] = {}
    for name, shift, bound in (
        ("d", d_shift, D_BOUND),
        ("s", s_shift, S_BOUND),
        ("h", h_shift, H_BOUND),
    ):
        gauge, solve_report = solve_gauge(
            source,
            target,
            shift.reshape(-1),
            edge_weight,
            len(valid_ids),
        )
        gauges[name] = gauge
        solver[name] = solve_report
        diagnostics[name] = edge_diagnostics(
            gauge, source, target, shift.reshape(-1), bound
        )

    gauge_d_shift = np.clip(
        np.rint(gauges["d"][:, None] - gauges["d"][neighbour_local]),
        -D_BOUND,
        D_BOUND,
    ).astype(np.int16)
    gauge_s_shift = np.clip(
        np.rint(gauges["s"][:, None] - gauges["s"][neighbour_local]),
        -S_BOUND,
        S_BOUND,
    ).astype(np.int16)
    gauge_h_shift = np.clip(
        np.rint(gauges["h"][:, None] - gauges["h"][neighbour_local]),
        -H_BOUND,
        H_BOUND,
    ).astype(np.int16)
    neighbour_sh = sh[neighbour_local]
    neighbour_d = d[neighbour_local]
    raw_sh = weighted_average(neighbour_sh, weight)
    raw_d = weighted_average(neighbour_d, weight)
    gauge_sh = weighted_average(
        v255.apply_2d_shifts(neighbour_sh, gauge_s_shift, gauge_h_shift), weight
    )
    gauge_d = weighted_average(
        v255.apply_1d_shifts(neighbour_d, gauge_d_shift), weight
    )
    oracle_sh = weighted_average(
        v255.apply_2d_shifts(neighbour_sh, s_shift, h_shift), weight
    )
    oracle_d = weighted_average(
        v255.apply_1d_shifts(neighbour_d, d_shift), weight
    )

    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Local official pair shifts are differences of smooth node-level propagation gauges, "
            "so graph synchronization should denoise them and recover much of the v255 oracle."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H pairwise shift labels",
                "official train coordinates",
                "train truth for an analysis-only synchronization upper bound",
            ],
            "does_not_use": [
                "synthetic or Sionna labels",
                "teacher output",
                "local replay truth",
                "hidden test labels",
                "raw RT H",
            ],
            "deployment": "none; all-node train gauges are an analysis object",
        },
        "parameters": {
            "valid_nodes": int(len(valid_ids)),
            "neighbours": NEIGHBORS,
            "power": POWER,
            "bounds": {"d": D_BOUND, "s": S_BOUND, "h": H_BOUND},
            "anchor_weight": ANCHOR_WEIGHT,
        },
        "solver": solver,
        "edge_diagnostics": diagnostics,
        "reconstruction": {
            "raw": metrics(raw_sh, raw_d, sh, d),
            "gauge_aligned": metrics(gauge_sh, gauge_d, sh, d),
            "pair_oracle": metrics(oracle_sh, oracle_d, sh, d),
        },
        "local_gate": None,
        "outputs": {"report": PATHS.relative(REPORT), "prediction": None, "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
