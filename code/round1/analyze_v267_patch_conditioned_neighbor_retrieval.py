#!/usr/bin/env python3
"""v267: nested-holdout patch-conditioned official train-H retrieval."""

from __future__ import annotations

import json

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

import analyze_v254_route_topology_spectral_field as v254
import analyze_v264_patch_scatterer_transformer as v264
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 267
VERSION = "v267_patch_conditioned_neighbor_retrieval"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
TRAIN_CANDIDATES = PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz"

OUTER_COUNT = 320
OUTER_SEED = 254
INNER_COUNT = 320
INNER_SEED = 267
PATCH_COUNT = 965
DELAY_BINS = 12
DELAY_BIN_WIDTH = 2.0
POWERS = (1.4, 2.0)
PATCH_LAMBDAS = (0.0, 2.0, 5.0, 9.0)
DELAY_LAMBDAS = (0.0, 2.0, 5.0, 9.0)
KS = (12, 24, 48)
EXACT_TOP = 12
ALPHAS = np.linspace(0.0, 1.0, 11, dtype=np.float32)


def normalize_l1(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 1e-12)
    out /= np.maximum(out.sum(axis=tuple(range(1, out.ndim)), keepdims=True), 1e-30)
    return out.astype(np.float32)


def normalize_l2(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-30)
    return out


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(normalize_l2(a) * normalize_l2(b), axis=1).astype(np.float32)


def sparse_row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    norm = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    scale = sparse.diags(1.0 / np.maximum(norm, 1e-12))
    return (scale @ matrix).tocsr()


def build_patch_signatures(candidates: dict[str, np.ndarray]) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    patch_ids = candidates["patch_id"][:, : v264.REFLECTIONS].astype(np.int64)
    score = candidates["score"][:, : v264.REFLECTIONS].astype(np.float32)
    path_length = candidates["path_length"][:, : v264.REFLECTIONS].astype(np.float32)
    delay = path_length / v264.LIGHT_SPEED * v264.S_SC * v264.SUBCARRIER_SPACING
    valid = score > 0.0
    rows = np.repeat(np.arange(len(score), dtype=np.int64), v264.REFLECTIONS)
    flat_valid = valid.ravel()
    rows = rows[flat_valid]
    ids = patch_ids.ravel()[flat_valid]
    values = np.sqrt(np.maximum(score.ravel()[flat_valid], 1e-6)).astype(np.float32)
    patch = sparse.coo_matrix(
        (values, (rows, ids)), shape=(len(score), PATCH_COUNT), dtype=np.float32
    ).tocsr()

    raw_bin = np.floor(delay.ravel()[flat_valid] / DELAY_BIN_WIDTH).astype(np.int64)
    raw_bin = np.clip(raw_bin, 0, DELAY_BINS - 1)
    pd_rows = []
    pd_cols = []
    pd_values = []
    for offset, multiplier in ((-1, 0.30), (0, 1.0), (1, 0.30)):
        current_bin = raw_bin + offset
        keep = (current_bin >= 0) & (current_bin < DELAY_BINS)
        pd_rows.append(rows[keep])
        pd_cols.append(ids[keep] * DELAY_BINS + current_bin[keep])
        pd_values.append(values[keep] * multiplier)
    patch_delay = sparse.coo_matrix(
        (
            np.concatenate(pd_values),
            (np.concatenate(pd_rows), np.concatenate(pd_cols)),
        ),
        shape=(len(score), PATCH_COUNT * DELAY_BINS),
        dtype=np.float32,
    ).tocsr()
    return sparse_row_normalize(patch), sparse_row_normalize(patch_delay)


def pair_similarity(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    patch: sparse.csr_matrix,
    patch_delay: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    patch_sim = (patch[query_ids] @ patch[bank_ids].T).toarray().astype(np.float32)
    delay_sim = (patch_delay[query_ids] @ patch_delay[bank_ids].T).toarray().astype(np.float32)
    return patch_sim, delay_sim


def kernel_neighbors(
    query_pos: np.ndarray,
    bank_pos: np.ndarray,
    patch_sim: np.ndarray,
    delay_sim: np.ndarray,
    power: float,
    patch_lambda: float,
    delay_lambda: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    distance = cdist(query_pos, bank_pos).astype(np.float32)
    logits = (
        -float(power) * np.log(np.maximum(distance, 0.05))
        + float(patch_lambda) * patch_sim
        + float(delay_lambda) * delay_sim
    )
    k = min(int(k), logits.shape[1])
    indices = np.argpartition(logits, -k, axis=1)[:, -k:]
    selected = np.take_along_axis(logits, indices, axis=1)
    order = np.argsort(selected, axis=1)[:, ::-1]
    indices = np.take_along_axis(indices, order, axis=1)
    selected = np.take_along_axis(selected, order, axis=1)
    selected -= selected.max(axis=1, keepdims=True)
    weights = np.exp(selected).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)
    return indices.astype(np.int64), weights


def weighted_target(bank: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    shape = (len(indices), indices.shape[1]) + (1,) * (bank.ndim - 1)
    return normalize_l1(np.sum(weights.reshape(shape) * bank[indices], axis=1))


def exact_prediction(
    bank_sh: np.ndarray,
    bank_d: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return weighted_target(bank_sh, indices, weights), weighted_target(bank_d, indices, weights)


def exact_joint(
    pred_sh: np.ndarray,
    pred_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> dict[str, float]:
    sh = float(np.mean(row_cos(pred_sh, truth_sh)))
    d = float(np.mean(row_cos(pred_d, truth_d)))
    return {"sh": sh, "d": d, "joint": 0.5 * (sh + d)}


def main() -> None:
    for path in (TARGET_CACHE, TRAIN_CANDIDATES):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    targets = np.load(TARGET_CACHE)
    sh = normalize_l1(targets["sh"].astype(np.float32))
    d = normalize_l1(targets["d"].astype(np.float32))
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    outer_local = v254.choose_seeded_matched_indices(
        train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, OUTER_SEED
    )
    outer_ids = np.intersect1d(outer_local, valid_ids)
    outer_bank_ids = np.setdiff1d(valid_ids, outer_ids)
    inner_local = v254.choose_seeded_matched_indices(
        train_pos[outer_bank_ids, :2], test_pos[:, :2], INNER_COUNT, INNER_SEED
    )
    inner_ids = outer_bank_ids[inner_local]
    inner_bank_ids = np.setdiff1d(outer_bank_ids, inner_ids)

    candidates = dict(np.load(TRAIN_CANDIDATES))
    patch, patch_delay = build_patch_signatures(candidates)
    inner_patch, inner_delay = pair_similarity(inner_ids, inner_bank_ids, patch, patch_delay)
    inner_sh_pair = normalize_l2(sh[inner_ids]) @ normalize_l2(sh[inner_bank_ids]).T
    inner_d_pair = normalize_l2(d[inner_ids]) @ normalize_l2(d[inner_bank_ids]).T
    inner_pair = 0.5 * (inner_sh_pair + inner_d_pair)

    approximate = []
    for power in POWERS:
        for patch_lambda in PATCH_LAMBDAS:
            for delay_lambda in DELAY_LAMBDAS:
                for k in KS:
                    indices, weights = kernel_neighbors(
                        train_pos[inner_ids, :2],
                        train_pos[inner_bank_ids, :2],
                        inner_patch,
                        inner_delay,
                        power,
                        patch_lambda,
                        delay_lambda,
                        k,
                    )
                    approx = float(
                        np.mean(np.sum(weights * np.take_along_axis(inner_pair, indices, axis=1), axis=1))
                    )
                    approximate.append(
                        {
                            "power": power,
                            "patch_lambda": patch_lambda,
                            "delay_lambda": delay_lambda,
                            "k": k,
                            "approx_joint": approx,
                        }
                    )
    approximate.sort(key=lambda item: float(item["approx_joint"]), reverse=True)

    inner_baseline_indices, inner_baseline_weights = kernel_neighbors(
        train_pos[inner_ids, :2],
        train_pos[inner_bank_ids, :2],
        np.zeros_like(inner_patch),
        np.zeros_like(inner_delay),
        2.0,
        0.0,
        0.0,
        16,
    )
    inner_base_sh, inner_base_d = exact_prediction(
        sh[inner_bank_ids], d[inner_bank_ids], inner_baseline_indices, inner_baseline_weights
    )
    inner_baseline = exact_joint(
        inner_base_sh, inner_base_d, sh[inner_ids], d[inner_ids]
    )

    exact_results = []
    seen = set()
    for spec in approximate:
        key = (spec["power"], spec["patch_lambda"], spec["delay_lambda"], spec["k"])
        if key in seen:
            continue
        seen.add(key)
        indices, weights = kernel_neighbors(
            train_pos[inner_ids, :2],
            train_pos[inner_bank_ids, :2],
            inner_patch,
            inner_delay,
            float(spec["power"]),
            float(spec["patch_lambda"]),
            float(spec["delay_lambda"]),
            int(spec["k"]),
        )
        pred_sh, pred_d = exact_prediction(
            sh[inner_bank_ids], d[inner_bank_ids], indices, weights
        )
        for alpha in ALPHAS:
            blend_sh = normalize_l1((1.0 - alpha) * inner_base_sh + alpha * pred_sh)
            blend_d = normalize_l1((1.0 - alpha) * inner_base_d + alpha * pred_d)
            metrics = exact_joint(blend_sh, blend_d, sh[inner_ids], d[inner_ids])
            exact_results.append({**spec, "alpha": float(alpha), **metrics})
        if len(seen) >= EXACT_TOP:
            break
    exact_results.sort(key=lambda item: float(item["joint"]), reverse=True)
    selected = exact_results[0]

    outer_patch, outer_delay = pair_similarity(outer_ids, outer_bank_ids, patch, patch_delay)
    outer_base_indices, outer_base_weights = kernel_neighbors(
        train_pos[outer_ids, :2],
        train_pos[outer_bank_ids, :2],
        np.zeros_like(outer_patch),
        np.zeros_like(outer_delay),
        2.0,
        0.0,
        0.0,
        16,
    )
    outer_patch_indices, outer_patch_weights = kernel_neighbors(
        train_pos[outer_ids, :2],
        train_pos[outer_bank_ids, :2],
        outer_patch,
        outer_delay,
        float(selected["power"]),
        float(selected["patch_lambda"]),
        float(selected["delay_lambda"]),
        int(selected["k"]),
    )
    outer_base_sh, outer_base_d = exact_prediction(
        sh[outer_bank_ids], d[outer_bank_ids], outer_base_indices, outer_base_weights
    )
    outer_patch_sh, outer_patch_d = exact_prediction(
        sh[outer_bank_ids], d[outer_bank_ids], outer_patch_indices, outer_patch_weights
    )
    alpha = float(selected["alpha"])
    outer_blend_sh = normalize_l1((1.0 - alpha) * outer_base_sh + alpha * outer_patch_sh)
    outer_blend_d = normalize_l1((1.0 - alpha) * outer_base_d + alpha * outer_patch_d)
    outer_baseline = exact_joint(
        outer_base_sh, outer_base_d, sh[outer_ids], d[outer_ids]
    )
    outer_patch_only = exact_joint(
        outer_patch_sh, outer_patch_d, sh[outer_ids], d[outer_ids]
    )
    outer_selected = exact_joint(
        outer_blend_sh, outer_blend_d, sh[outer_ids], d[outer_ids]
    )

    selected_neighbor_patch = np.take_along_axis(outer_patch, outer_patch_indices, axis=1)
    selected_neighbor_delay = np.take_along_axis(outer_delay, outer_patch_indices, axis=1)
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Official train points sharing stable wall-patch and patch-delay signatures are "
            "better spectral neighbors than points selected by Euclidean distance alone."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H S-H/PDP targets",
                "official train/test coordinates",
                "official point-cloud patch IDs, scores, and analytic path delays",
                "nested official holdout selection",
            ],
            "does_not_use": [
                "Sionna or synthetic labels",
                "teacher output",
                "local replay truth",
                "hidden labels",
                "raw RT H",
            ],
        },
        "parameters": {
            "outer_fit_count": int(len(outer_bank_ids)),
            "outer_validation_count": int(len(outer_ids)),
            "inner_fit_count": int(len(inner_bank_ids)),
            "inner_validation_count": int(len(inner_ids)),
            "grid_count": int(len(approximate)),
            "exact_top": EXACT_TOP,
        },
        "inner": {
            "baseline": inner_baseline,
            "selected": selected,
            "gain": float(selected["joint"] - inner_baseline["joint"]),
            "top_approximate": approximate[:20],
            "top_exact": exact_results[:20],
        },
        "outer": {
            "baseline": outer_baseline,
            "patch_only": outer_patch_only,
            "selected": outer_selected,
            "gain": float(outer_selected["joint"] - outer_baseline["joint"]),
            "sh_gain": float(outer_selected["sh"] - outer_baseline["sh"]),
            "d_gain": float(outer_selected["d"] - outer_baseline["d"]),
            "selected_neighbor_patch_cos": summarize(selected_neighbor_patch),
            "selected_neighbor_patch_delay_cos": summarize(selected_neighbor_delay),
        },
        "outputs": {"report": PATHS.relative(REPORT), "prediction": None, "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
