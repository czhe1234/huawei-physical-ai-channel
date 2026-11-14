#!/usr/bin/env python3
"""Anchor-bounded PDP decoder for v291 high-depth geometry-only paths."""

from __future__ import annotations

import json
import random

import numpy as np
import torch
from scipy.spatial import cKDTree

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import analyze_v252_pointcloud_pathfield_calibrator as v252
import analyze_v254_route_topology_spectral_field as v254
import analyze_v257_neural_pairwise_spectral_transport as v257
import generate_v215_pointcloud_physics_neural_field as v215
from generate_v87_easy_field_projection import summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 292
VERSION = "v292_highdepth_path_weight_transformer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v292_highdepth_pdp_targets.npz"
OUTPUT = OUT_DIR / "v292_Round1_Test_Channel.npy"
PATH_CACHE = PATHS.cache_dir(291) / "v291_highdepth_pointcloud_paths.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

LOCAL_GATE = 0.645604
NEIGHBORS = 16
EPOCHS = 60
OUTER_SEEDS = (292, 1292, 2292, 3292, 4292)
OUTER_COUNT = 320
INNER_COUNT = 240
FINAL_SEEDS = (29201, 29231, 29261)
MIN_OFFICIAL_DELTA = 5e-4
ALPHA_GRID = np.linspace(0.0, 1.5, 13, dtype=np.float32)
MAX_DEPTH = 16


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    out /= np.maximum(out.sum(axis=1, keepdims=True), TINY)
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def knn_anchor(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    query_pos: np.ndarray,
    bank_pos: np.ndarray,
    bank_d: np.ndarray,
    exclude_self: bool,
) -> np.ndarray:
    extra = 1 if exclude_self else 0
    distance, local = cKDTree(bank_pos[bank_ids, :2]).query(
        query_pos[query_ids, :2], k=NEIGHBORS + extra
    )
    if distance.ndim == 1:
        distance = distance[:, None]
        local = local[:, None]
    absolute_all = bank_ids[local]
    if exclude_self:
        absolute = np.empty((len(query_ids), NEIGHBORS), dtype=np.int64)
        kept_distance = np.empty((len(query_ids), NEIGHBORS), dtype=np.float64)
        for row, query_id in enumerate(query_ids):
            keep = absolute_all[row] != query_id
            absolute[row] = absolute_all[row, keep][:NEIGHBORS]
            kept_distance[row] = distance[row, keep][:NEIGHBORS]
        distance = kept_distance
    else:
        absolute = absolute_all
    weight = 1.0 / np.maximum(distance, 0.05) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return normalize(
        np.sum(weight[:, :, None] * bank_d[absolute], axis=1, dtype=np.float64)
    )


def geometry_features(
    cache: np.lib.npyio.NpzFile,
    prefix: str,
    ids: np.ndarray,
    node: np.ndarray,
    anchor: np.ndarray,
) -> v252.Features:
    def take(name: str) -> np.ndarray:
        return cache[f"{prefix}_{name}"][ids]

    valid = take("valid").astype(bool)
    delay = np.nan_to_num(take("delay_bin_physical").astype(np.float32), nan=0.0)
    length = np.nan_to_num(
        take("path_length").astype(np.float32), nan=0.0, posinf=1000.0
    )
    theta_t = np.nan_to_num(take("theta_t").astype(np.float32), nan=0.0)
    phi_t = np.nan_to_num(take("phi_t").astype(np.float32), nan=0.0)
    theta_r = np.nan_to_num(take("theta_r").astype(np.float32), nan=0.0)
    phi_r = np.nan_to_num(take("phi_r").astype(np.float32), nan=0.0)
    depth = take("interaction_count").astype(np.float32)
    mask = take("interaction_mask").astype(np.int16)
    score = np.nan_to_num(
        take("score").astype(np.float32), nan=-20.0, neginf=-20.0, posinf=8.0
    )
    interactions = take("interactions").astype(np.int16)
    interaction_hist = np.stack(
        [
            np.mean(interactions == value, axis=2, dtype=np.float32)
            for value in (1, 2, 4, 8)
        ],
        axis=2,
    )
    bits = np.stack(
        [((mask & bit) != 0).astype(np.float32) for bit in (1, 2, 4, 8)],
        axis=2,
    )
    prior = score - 0.10 * depth - 0.003 * np.maximum(length - 50.0, 0.0)
    maximum = np.max(np.where(valid, prior, -100.0), axis=1, keepdims=True)
    relative_prior = np.where(valid, np.clip(prior - maximum, -24.0, 0.0), 0.0)
    numeric = np.concatenate(
        [
            valid[..., None].astype(np.float32),
            (delay / 32.0)[..., None],
            (np.log1p(length) / 7.0)[..., None],
            (relative_prior / 12.0)[..., None],
            (score / 6.0)[..., None],
            (depth / MAX_DEPTH)[..., None],
            bits,
            interaction_hist,
            np.stack(
                [
                    np.sin(theta_t),
                    np.cos(theta_t),
                    np.sin(phi_t),
                    np.cos(phi_t),
                    np.sin(theta_r),
                    np.cos(theta_r),
                    np.sin(phi_r),
                    np.cos(phi_r),
                ],
                axis=2,
            ),
        ],
        axis=2,
    ).astype(np.float32)
    type_id = np.zeros_like(mask, dtype=np.int64)
    type_id[valid & (depth == 0)] = 1
    type_id[valid & ((mask & 8) != 0)] = 5
    type_id[valid & ((mask & 4) != 0) & (type_id == 0)] = 4
    type_id[valid & ((mask & 2) != 0) & (type_id == 0)] = 3
    type_id[valid & ((mask & 1) != 0) & (type_id == 0)] = 2
    type_id[valid & (type_id == 0)] = 6

    depth_hist = np.stack(
        [np.mean(valid & (depth == value), axis=1) for value in range(MAX_DEPTH + 1)],
        axis=1,
    ).astype(np.float32)
    global_features = np.concatenate(
        [
            node[ids].astype(np.float32),
            np.log(np.maximum(anchor, 1e-11)).astype(np.float32),
            v215.d_stats(anchor),
            valid.mean(axis=1, keepdims=True).astype(np.float32),
            (np.max(depth, axis=1, keepdims=True) / MAX_DEPTH).astype(np.float32),
            depth_hist,
        ],
        axis=1,
    ).astype(np.float32)
    return v252.Features(
        np.nan_to_num(global_features, nan=0.0, posinf=8.0, neginf=-8.0),
        np.nan_to_num(numeric, nan=0.0, posinf=8.0, neginf=-8.0),
        type_id,
        valid,
        delay,
        relative_prior.astype(np.float32),
    )


def choose_alpha(
    base: np.ndarray,
    model: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    curve = []
    for alpha in ALPHA_GRID:
        value = normalize(base + float(alpha) * (model - base))
        curve.append(
            {
                "alpha": float(alpha),
                "cosine": float(np.mean(row_cos(value, truth))),
            }
        )
    best = max(curve, key=lambda value: value["cosine"])
    return float(best["alpha"]), curve


def main() -> None:
    for path in (PATH_CACHE, TARGET_CACHE, BASE, GEOM_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    set_seed(VERSION_NUMBER)
    path_cache = np.load(PATH_CACHE)
    target_data = np.load(TARGET_CACHE)
    truth_d = normalize(target_data["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    train_node, test_node = v257.build_node_features(
        train_pos[:, :2], test_pos[:, :2]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"v292 device={device} valid={len(valid_ids)}", flush=True)

    outer_rows: list[dict[str, object]] = []
    oof_ids_parts: list[np.ndarray] = []
    oof_base_parts: list[np.ndarray] = []
    oof_model_parts: list[np.ndarray] = []
    selected_epochs: list[int] = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], INNER_COUNT, seed + 50_000
            ),
            outer_fit,
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)

        inner_train_anchor = knn_anchor(
            inner_fit, inner_fit, train_pos, train_pos, truth_d, True
        )
        inner_val_anchor = knn_anchor(
            inner_ids, inner_fit, train_pos, train_pos, truth_d, False
        )
        inner_train_raw = geometry_features(
            path_cache, "train", inner_fit, train_node, inner_train_anchor
        )
        inner_val_raw = geometry_features(
            path_cache, "train", inner_ids, train_node, inner_val_anchor
        )
        inner_train, [inner_val], _ = v252.standardize(
            inner_train_raw, [inner_val_raw]
        )
        _, inner_training = v252.train_model(
            inner_train,
            inner_train_anchor,
            truth_d[inner_fit],
            seed,
            EPOCHS,
            device,
            (inner_val, inner_val_anchor, truth_d[inner_ids]),
        )
        selected_epoch = int(inner_training["best_epoch"])
        selected_epochs.append(selected_epoch)

        outer_train_anchor = knn_anchor(
            outer_fit, outer_fit, train_pos, train_pos, truth_d, True
        )
        outer_val_anchor = knn_anchor(
            outer_ids, outer_fit, train_pos, train_pos, truth_d, False
        )
        outer_train_raw = geometry_features(
            path_cache, "train", outer_fit, train_node, outer_train_anchor
        )
        outer_val_raw = geometry_features(
            path_cache, "train", outer_ids, train_node, outer_val_anchor
        )
        outer_train, [outer_val], _ = v252.standardize(
            outer_train_raw, [outer_val_raw]
        )
        model, outer_training = v252.train_model(
            outer_train,
            outer_train_anchor,
            truth_d[outer_fit],
            seed + 100_000,
            selected_epoch,
            device,
            None,
        )
        model_d, path_d, gate = v252.predict(
            model, outer_val, outer_val_anchor, device
        )
        baseline = float(np.mean(row_cos(outer_val_anchor, truth_d[outer_ids])))
        model_score = float(np.mean(row_cos(model_d, truth_d[outer_ids])))
        outer_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "fit_count": int(len(outer_fit)),
                "validation_count": int(len(outer_ids)),
                "selected_epoch": selected_epoch,
                "inner_training": inner_training,
                "outer_training": outer_training,
                "baseline": baseline,
                "model": model_score,
                "raw_delta": model_score - baseline,
                "path_cosine": float(np.mean(row_cos(path_d, truth_d[outer_ids]))),
                "gate": summarize(gate),
            }
        )
        oof_ids_parts.append(outer_ids)
        oof_base_parts.append(outer_val_anchor)
        oof_model_parts.append(model_d)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"v292 outer {fold + 1}/5 epoch={selected_epoch} "
            f"base={baseline:.6f} model={model_score:.6f} "
            f"delta={model_score - baseline:+.6f}",
            flush=True,
        )

    oof_ids = np.concatenate(oof_ids_parts)
    oof_base = np.concatenate(oof_base_parts)
    oof_model = np.concatenate(oof_model_parts)
    alpha, alpha_curve = choose_alpha(
        oof_base, oof_model, truth_d[oof_ids]
    )
    fold_delta: list[float] = []
    for row, ids, base, model in zip(
        outer_rows, oof_ids_parts, oof_base_parts, oof_model_parts
    ):
        blended = normalize(base + alpha * (model - base))
        score = float(np.mean(row_cos(blended, truth_d[ids])))
        delta = score - float(row["baseline"])
        row["blended"] = score
        row["blended_delta"] = delta
        fold_delta.append(delta)
    mean_delta = float(np.mean(fold_delta))
    wins = int(np.sum(np.asarray(fold_delta) > 0.0))
    official_selected = bool(
        alpha > 0.0 and mean_delta >= MIN_OFFICIAL_DELTA and wins >= 3
    )

    final_training: list[dict[str, object]] = []
    local_score = None
    projection = None
    test_gate = None
    if official_selected:
        final_epoch = max(int(np.median(selected_epochs)), 1)
        final_anchor = knn_anchor(
            valid_ids, valid_ids, train_pos, train_pos, truth_d, True
        )
        test_ids = np.arange(len(test_pos), dtype=np.int64)
        test_anchor = knn_anchor(
            test_ids, valid_ids, test_pos, train_pos, truth_d, False
        )
        final_raw = geometry_features(
            path_cache, "train", valid_ids, train_node, final_anchor
        )
        test_raw = geometry_features(
            path_cache, "test", test_ids, test_node, test_anchor
        )
        final_features, [test_features], feature_stats = v252.standardize(
            final_raw, [test_raw]
        )
        model_parts: list[np.ndarray] = []
        gate_parts: list[np.ndarray] = []
        for seed in FINAL_SEEDS:
            model, training = v252.train_model(
                final_features,
                final_anchor,
                truth_d[valid_ids],
                seed,
                final_epoch,
                device,
                None,
            )
            model_d, _path_d, gate = v252.predict(
                model, test_features, test_anchor, device
            )
            model_parts.append(model_d)
            gate_parts.append(gate)
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(
                {
                    "version": VERSION,
                    "seed": seed,
                    "epoch": final_epoch,
                    "state_dict": model.state_dict(),
                    "feature_stats": feature_stats,
                },
                checkpoint,
            )
            final_training.append(
                {**training, "checkpoint": PATHS.relative(checkpoint)}
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        test_model = normalize(np.mean(model_parts, axis=0))
        test_gate = np.mean(gate_parts, axis=0).astype(np.float32)
        current = v225.current_delay_marginals(BASE)
        residual = test_model - test_anchor
        target_d = normalize(current["d"] + alpha * residual)
        regimes = regime_of(
            np.load(GEOM_CACHE)["test_blocked"].astype(np.float32)
        ).astype(str)
        np.savez_compressed(
            TARGETS_OUT,
            anchor_d=test_anchor,
            model_d=test_model,
            current_d=current["d"],
            target_d=target_d,
            residual=residual,
            alpha=np.asarray(alpha, dtype=np.float32),
            gate=test_gate,
        )
        projection = v225.apply_delay_targets(
            BASE,
            OUTPUT,
            {"d": target_d, "du": current["du"], "dh": current["dh"]},
            regimes,
            1.05,
        )
        local_score = v208.score_prediction(OUTPUT)

    report = {
        "method": VERSION,
        "status": (
            "local_gate_passed_needs_packaging"
            if local_score is not None
            and float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "Depth-16 interaction geometry supplies propagation-state information that lets "
            "an anchor-bounded Transformer predict high-order path weights better than a "
            "depth-agnostic official PDP KNN anchor."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H PDP",
                "official train/test coordinates and point-cloud geometry",
                "v291 depth-16 geometry/visibility/interactions without coefficients",
                "PyTorch path Transformer trained only on official train folds",
                "strict v239 official-only carrier after official selection",
            ],
            "does_not_use": [
                "Paths.cfr or path coefficients",
                "synthetic/Sionna channel labels",
                "v212/v217 teacher output",
                "hidden labels or local truth for training/selection",
            ],
            "local_truth_usage": (
                "one final gate score after nested official selection"
                if official_selected
                else "not read"
            ),
        },
        "parameters": {
            "neighbors": NEIGHBORS,
            "epochs": EPOCHS,
            "outer_seeds": list(OUTER_SEEDS),
            "outer_count": OUTER_COUNT,
            "inner_count": INNER_COUNT,
            "alpha_grid": ALPHA_GRID.tolist(),
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
            "final_seeds": list(FINAL_SEEDS),
        },
        "outer_folds": outer_rows,
        "official_selection": {
            "residual_alpha": alpha,
            "alpha_curve": alpha_curve,
            "mean_pdp_delta": mean_delta,
            "wins": wins,
            "selected_epochs": selected_epochs,
            "selected": official_selected,
        },
        "final_training": final_training,
        "test_gate": summarize(test_gate) if test_gate is not None else None,
        "projection": (
            {key: summarize(value) for key, value in projection.items()}
            if projection is not None
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
            "targets": PATHS.relative(TARGETS_OUT) if TARGETS_OUT.is_file() else None,
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
