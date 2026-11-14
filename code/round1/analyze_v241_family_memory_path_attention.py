#!/usr/bin/env python3
"""v241: reflection-family-conditioned path-memory kernel attention.

Each query path attends only to official train paths from the same reflection
family, then transfers NNLS path-energy labels with a geometry kernel.  Kernel
size and exponent are selected on the official spatial validation split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import analyze_v233_official_joint_ad_manifold as v233
import analyze_v238_raw_path_conditional_angle as v238
import analyze_v239_nnls_path_energy_decoder as v239
import generate_v215_pointcloud_physics_neural_field as v215
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 241
VERSION = "v241_family_memory_path_attention"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / "v241_family_memory_path_attention.json"
TARGET_OUT = OUT_DIR / "v241_test_family_memory_pdp.npy"
ACTION_OUT = OUT_DIR / "v241_family_memory_actions.npz"
BASE_V239 = v239.OUT_DIR / "v239_path_on_v237_Round1_Test_Channel.npy"
K_GRID = (4, 8, 16, 32, 64)
POWER_GRID = (0.5, 1.0, 2.0)
MAX_K = max(K_GRID)


def path_feature(prefix: str, pos: np.ndarray, indices: np.ndarray | None) -> np.ndarray:
    cache = np.load(v215.PATH_CACHE)

    def take(name: str) -> np.ndarray:
        value = cache[f"{prefix}_{name}"]
        return value if indices is None else value[indices]

    count = len(pos)
    paths = take("valid").shape[1]
    ue = np.repeat(pos[:, None, :], paths, axis=1)
    values = [
        ue / np.asarray([120.0, 120.0, 30.0], dtype=np.float32),
        np.nan_to_num(take("delay_bin")[..., None], nan=0.0) / 10.0,
        np.nan_to_num(take("score")[..., None], nan=-4.0, posinf=4.0, neginf=-4.0) / 4.0,
        np.nan_to_num(take("incidence_cos")[..., None], nan=0.0),
        np.nan_to_num(take("excess_length")[..., None], nan=300.0) / 300.0,
        np.nan_to_num(take("reflect_point"), nan=0.0) / 300.0,
        np.nan_to_num(take("reflect_point_2"), nan=0.0) / 300.0,
        np.nan_to_num(take("tx_dir"), nan=0.0),
        np.nan_to_num(take("rx_dir"), nan=0.0),
    ]
    out = np.concatenate(values, axis=2).astype(np.float32)
    return np.nan_to_num(out.reshape(count, paths, -1), nan=0.0, posinf=4.0, neginf=-4.0)


def family(prefix: str, indices: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = np.load(v215.PATH_CACHE)

    def take(name: str) -> np.ndarray:
        value = cache[f"{prefix}_{name}"].astype(np.int64)
        return value if indices is None else value[indices]

    return take("path_type"), take("plane_id"), take("plane_id_2")


@dataclass
class Memory:
    path_labels: np.ndarray
    path_dist: np.ndarray
    path_count: np.ndarray
    fallback_level: np.ndarray
    tail_labels: np.ndarray
    tail_dist: np.ndarray


def build_memory(
    bank_pos: np.ndarray,
    query_pos: np.ndarray,
    bank_feature: np.ndarray,
    query_feature: np.ndarray,
    bank_family: tuple[np.ndarray, np.ndarray, np.ndarray],
    query_family: tuple[np.ndarray, np.ndarray, np.ndarray],
    bank_weights: np.ndarray,
    bank_blocked: np.ndarray,
    query_blocked: np.ndarray,
) -> Memory:
    mean = bank_feature.reshape(-1, bank_feature.shape[-1]).mean(axis=0)
    std = np.maximum(bank_feature.reshape(-1, bank_feature.shape[-1]).std(axis=0), 0.08)
    bank_x = np.clip((bank_feature - mean) / std, -8.0, 8.0)
    query_x = np.clip((query_feature - mean) / std, -8.0, 8.0)
    bt, bp1, bp2 = bank_family
    qt, qp1, qp2 = query_family
    exact: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    first: dict[tuple[int, int], list[tuple[int, int]]] = {}
    typed: dict[int, list[tuple[int, int]]] = {}
    for i in range(len(bank_pos)):
        for j in range(bt.shape[1]):
            item = (i, j)
            exact.setdefault((int(bt[i, j]), int(bp1[i, j]), int(bp2[i, j])), []).append(item)
            first.setdefault((int(bt[i, j]), int(bp1[i, j])), []).append(item)
            typed.setdefault(int(bt[i, j]), []).append(item)

    path_labels = np.zeros((len(query_pos), qt.shape[1], MAX_K, len(v239.SHIFTS)), dtype=np.float32)
    path_dist = np.full((len(query_pos), qt.shape[1], MAX_K), np.inf, dtype=np.float32)
    path_count = np.zeros((len(query_pos), qt.shape[1]), dtype=np.int32)
    fallback = np.full((len(query_pos), qt.shape[1]), 2, dtype=np.int8)
    reshaped = bank_weights[:, :-v239.TAIL_COUNT].reshape(len(bank_pos), qt.shape[1], len(v239.SHIFTS))
    for i in range(len(query_pos)):
        for j in range(qt.shape[1]):
            candidates = exact.get((int(qt[i, j]), int(qp1[i, j]), int(qp2[i, j])))
            level = 0
            if not candidates:
                candidates = first.get((int(qt[i, j]), int(qp1[i, j])))
                level = 1
            if not candidates:
                candidates = typed[int(qt[i, j])]
                level = 2
            bi = np.asarray([item[0] for item in candidates], dtype=np.int64)
            bj = np.asarray([item[1] for item in candidates], dtype=np.int64)
            dist = np.sum((bank_x[bi, bj] - query_x[i, j]) ** 2, axis=1)
            take = np.argsort(dist)[: min(MAX_K, len(dist))]
            count = len(take)
            path_labels[i, j, :count] = reshaped[bi[take], bj[take]]
            path_dist[i, j, :count] = np.sqrt(np.maximum(dist[take], 0.0)).astype(np.float32)
            path_count[i, j] = count
            fallback[i, j] = level

    bank_global = np.concatenate(
        [
            bank_pos / np.asarray([120.0, 120.0, 30.0], dtype=np.float32),
            bank_blocked[:, None] / 160.0,
            np.mean(bank_feature, axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    query_global = np.concatenate(
        [
            query_pos / np.asarray([120.0, 120.0, 30.0], dtype=np.float32),
            query_blocked[:, None] / 160.0,
            np.mean(query_feature, axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    gmean = bank_global.mean(axis=0)
    gstd = np.maximum(bank_global.std(axis=0), 0.08)
    bank_global = np.clip((bank_global - gmean) / gstd, -8.0, 8.0)
    query_global = np.clip((query_global - gmean) / gstd, -8.0, 8.0)
    dist = np.maximum(
        np.sum(query_global**2, axis=1, keepdims=True)
        + np.sum(bank_global**2, axis=1)[None, :]
        - 2.0 * query_global @ bank_global.T,
        0.0,
    )
    idx = np.argpartition(dist, min(MAX_K, len(bank_pos) - 1), axis=1)[:, :MAX_K]
    vals = np.take_along_axis(dist, idx, axis=1)
    order = np.argsort(vals, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    tail_labels = bank_weights[idx, -v239.TAIL_COUNT:].astype(np.float32)
    tail_dist = np.sqrt(np.maximum(vals, 0.0)).astype(np.float32)
    return Memory(path_labels, path_dist, path_count, fallback, tail_labels, tail_dist)


def kernel_average(labels: np.ndarray, distance: np.ndarray, k: int, power: float) -> np.ndarray:
    value = labels[..., :k, :]
    dist = distance[..., :k]
    valid = np.isfinite(dist)
    weight = np.where(valid, 1.0 / np.maximum(dist, 0.08) ** power, 0.0).astype(np.float32)
    weight /= np.maximum(weight.sum(axis=-1, keepdims=True), 1e-30)
    return np.sum(weight[..., None] * value, axis=-2).astype(np.float32)


def predict_weights(memory: Memory, k: int, power: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    path = kernel_average(memory.path_labels, memory.path_dist, k, power)
    tail = kernel_average(memory.tail_labels, memory.tail_dist, k, power)
    out = np.concatenate([path.reshape(len(path), -1), tail], axis=1)
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-30)
    nearest = np.mean(memory.path_dist[:, :, 0], axis=1).astype(np.float32)
    exact_fraction = np.mean(memory.fallback_level == 0, axis=1).astype(np.float32)
    confidence = (
        exact_fraction
        * np.exp(-0.20 * np.clip(nearest, 0.0, 12.0))
        * np.clip(np.mean(memory.path_count, axis=1) / max(float(k), 1.0), 0.0, 1.0)
    ).astype(np.float32)
    return out.astype(np.float32), {
        "nearest": nearest,
        "exact_fraction": exact_fraction,
        "confidence": confidence,
    }


def reconstruct(weights: np.ndarray, prefix: str, indices: np.ndarray | None) -> np.ndarray:
    cache = np.load(v215.PATH_CACHE)
    delay = cache[f"{prefix}_delay_bin"].astype(np.float32)
    path_type = cache[f"{prefix}_path_type"].astype(np.int64)
    valid = cache[f"{prefix}_valid"].astype(bool)
    if indices is not None:
        delay = delay[indices]
        path_type = path_type[indices]
        valid = valid[indices]
    atoms = v239.numpy_atoms(delay, path_type, valid)
    d = np.sum(weights[..., None] * atoms, axis=1)
    return v239.normalize_d(d)


def run_variant(
    name: str,
    source,
    predicted_d: np.ndarray,
    alpha: np.ndarray,
    regimes: np.ndarray,
    clip: float,
) -> dict[str, object]:
    current = v225.current_delay_marginals(source)
    target_d = v239.normalize_d((1.0 - alpha[:, None]) * current["d"] + alpha[:, None] * predicted_d)
    targets = {"d": target_d, "du": current["du"], "dh": current["dh"]}
    output = OUT_DIR / f"v241_{name}_Round1_Test_Channel.npy"
    diag = v225.apply_delay_targets(source, output, targets, regimes, clip)
    return {
        "name": name,
        "source": PATHS.relative(source),
        "output": PATHS.relative(output),
        "score": v208.score_prediction(output),
        "alpha": summarize(alpha),
        "target_l1": summarize(v225.l1_rows(target_d, current["d"])),
        "diagnostics": {key: summarize(value) for key, value in diag.items()},
    }


def main() -> None:
    for path in [v239.BASE_V237, BASE_V239, v215.PATH_CACHE, v239.NNLS_CACHE, v238.GEOM_CACHE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    clean_targets, _raw_targets, _log_power, valid_delay = v215.load_targets()
    target_all = v239.normalize_d(clean_targets["d"])
    _support_raw, valid_support, log_power = v238.build_raw_support()
    valid = valid_delay & valid_support
    valid_ids = np.flatnonzero(valid).astype(np.int64)
    nnls_weights = np.load(v239.NNLS_CACHE)["weights"].astype(np.float32)
    support = np.load(v238.OUT_DIR / "v238_train_raw_angle_delay_normalized.npy", mmap_mode="r")
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(v238.GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    bundle = v238.build_bundle(support, valid_ids, log_power, train_pos, test_pos, train_blocked, test_blocked)
    train_ids = bundle["train_ids"]
    val_ids = bundle["val_ids"]

    train_feature = path_feature("train", train_pos[train_ids], train_ids)
    val_feature = path_feature("train", train_pos[val_ids], val_ids)
    train_family = family("train", train_ids)
    val_family = family("train", val_ids)
    val_memory = build_memory(
        train_pos[train_ids],
        train_pos[val_ids],
        train_feature,
        val_feature,
        train_family,
        val_family,
        nnls_weights[train_ids],
        train_blocked[train_ids],
        train_blocked[val_ids],
    )
    grid = []
    for k in K_GRID:
        for power in POWER_GRID:
            weight, diag = predict_weights(val_memory, k, power)
            pred = reconstruct(weight, "train", val_ids)
            score = float(np.mean(v239.row_cos(pred, target_all[val_ids])))
            grid.append(
                {
                    "k": k,
                    "power": power,
                    "cos": score,
                    "confidence": summarize(diag["confidence"]),
                }
            )
    selected = max(grid, key=lambda item: item["cos"])
    val_weights, val_diag = predict_weights(val_memory, int(selected["k"]), float(selected["power"]))
    val_d = reconstruct(val_weights, "train", val_ids)
    val_anchor_d = v239.normalize_d(np.asarray(bundle["val_anchor"]).sum(axis=(1, 2)))
    val_alpha, calibration = v239.calibrate_alpha(
        val_anchor_d, val_d, target_all[val_ids], regime_of(train_blocked[val_ids])
    )

    final_feature = path_feature("train", train_pos[valid_ids], valid_ids)
    test_feature = path_feature("test", test_pos, None)
    final_family = family("train", valid_ids)
    test_family = family("test", None)
    test_memory = build_memory(
        train_pos[valid_ids],
        test_pos,
        final_feature,
        test_feature,
        final_family,
        test_family,
        nnls_weights[valid_ids],
        train_blocked[valid_ids],
        test_blocked,
    )
    test_weights, test_diag = predict_weights(test_memory, int(selected["k"]), float(selected["power"]))
    test_d = reconstruct(test_weights, "test", None)
    np.save(TARGET_OUT, test_d)
    regimes = regime_of(test_blocked)
    confidence = np.clip(test_diag["confidence"] / 0.72, 0.0, 1.0).astype(np.float32)
    alpha_base = np.asarray([val_alpha[str(regime)] for regime in regimes], dtype=np.float32)
    variants = [
        (
            "family_on_v237",
            v239.BASE_V237,
            np.clip(
                np.asarray([{"clear": 0.0, "light": 0.04, "medium": 0.22, "heavy": 0.38}[str(r)] for r in regimes], dtype=np.float32)
                * confidence,
                0.0,
                0.42,
            ),
            1.04,
        ),
        (
            "family_on_v239",
            BASE_V239,
            np.clip(
                np.asarray([{"clear": 0.0, "light": 0.025, "medium": 0.14, "heavy": 0.24}[str(r)] for r in regimes], dtype=np.float32)
                * confidence,
                0.0,
                0.28,
            ),
            1.03,
        ),
        (
            "calibrated_on_v237",
            v239.BASE_V237,
            np.clip(alpha_base * (0.35 + 0.65 * confidence), 0.0, 0.92),
            1.04,
        ),
        (
            "family_heavy_overdrive",
            v239.BASE_V237,
            np.clip(
                np.asarray([{"clear": 0.0, "light": 0.02, "medium": 0.32, "heavy": 0.56}[str(r)] for r in regimes], dtype=np.float32)
                * (0.42 + 0.58 * confidence),
                0.0,
                0.62,
            ),
            1.10,
        ),
    ]
    results = [run_variant(name, source, test_d, alpha.astype(np.float32), regimes, clip) for name, source, alpha, clip in variants]
    best = max(results, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(
        ACTION_OUT,
        weights=test_weights,
        nearest=test_diag["nearest"],
        exact_fraction=test_diag["exact_fraction"],
        confidence=confidence,
        predicted_vs_anchor=v239.row_cos(
            test_d, v239.normalize_d(np.asarray(bundle["test_anchor"]).sum(axis=(1, 2)))
        ),
        predicted_vs_v239=v239.row_cos(test_d, v225.current_delay_marginals(BASE_V239)["d"]),
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": "Path energy is locally transferable within an exact reflection family even when a global geometry-to-weight Transformer averages incompatible families.",
        "compliance_boundary": {
            "generation_uses": [
                "official train H NNLS path-energy labels",
                "official train/test coordinates",
                "official point-cloud plane IDs and path geometry",
                "v237/v239 strict official-data carriers",
            ],
            "generation_does_not_use": [
                "Sionna generated channel labels",
                "v212/v217/v221/v222 teacher outputs",
                "synthetic replay truth",
                "hidden online labels",
                "raw RT H",
            ],
        },
        "model_class": "reflection-family-conditioned kNN/kernel attention",
        "official_validation": {
            "grid": grid,
            "selected": selected,
            "anchor_cos": float(np.mean(v239.row_cos(val_anchor_d, target_all[val_ids]))),
            "kernel_cos": float(np.mean(v239.row_cos(val_d, target_all[val_ids]))),
            "calibration": calibration,
            "diagnostics": {key: summarize(value) for key, value in val_diag.items()},
        },
        "test_diagnostics": {
            "nearest": summarize(test_diag["nearest"]),
            "exact_fraction": summarize(test_diag["exact_fraction"]),
            "confidence": summarize(confidence),
            "predicted_vs_anchor": summarize(
                v239.row_cos(test_d, v239.normalize_d(np.asarray(bundle["test_anchor"]).sum(axis=(1, 2))))
            ),
            "predicted_vs_v239": summarize(v239.row_cos(test_d, v225.current_delay_marginals(BASE_V239)["d"])),
        },
        "variants": results,
        "local_gate": {
            "required_score_traceback_leaked": v239.GATE_SCORE,
            "best_variant": best["name"],
            "best_score_traceback_leaked": best["score"]["score_traceback_leaked"],
            "gap_to_gate": best["score"]["score_traceback_leaked"] - v239.GATE_SCORE,
        },
        "outputs": {
            "target": PATHS.relative(TARGET_OUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
