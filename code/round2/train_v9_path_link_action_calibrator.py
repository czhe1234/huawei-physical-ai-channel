#!/usr/bin/env python3
"""Cross-fit a lightweight action calibrator on top of the 4000-point path gate."""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import torch

from analyze_joint_path_token_oracle import OOF_META
from analyze_v4_exact_joint_projection import truth_views
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics
from train_full_path_pdp_link_gate import POINT_CONTEXT, V9_ALPHA
from train_path_pdp_link_gate import link_coordinates, project


VERSION = "Round2-E67-V9-path-link-action-calibrator"
E65_DIR = PATHS.artifacts / "round2" / "path_pdp_link_gate"
PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n469_s24_k128_b096_d064_scatter1_sel106"
)
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v9_path_link_action_calibrator"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v9_path_link_action_calibrator"
FEATURES = ARTIFACT_DIR / "link_features_f16.npy"
BEST_ALPHA = ARTIFACT_DIR / "best_alpha_f16.npy"
BEST_GAIN = ARTIFACT_DIR / "best_gain_f16.npy"
SCORE_GRID = ARTIFACT_DIR / "score_grid_f16.npy"
OOF_PREDICTION = ARTIFACT_DIR / "oof_prediction.npz"
REPORT = ANALYSIS_DIR / "v9_path_link_action_calibrator.json"
ALPHA_GRID = np.asarray([0.0, 0.10, 0.25, 0.40, 0.60, 0.80, 1.0], dtype=np.float32)
FEATURE_COUNT = 49
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=260_000)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--random-state", type=int, default=2026080667)
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def normalize(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = value.sum(axis=-1, keepdims=True, dtype=np.float64)
    return np.divide(
        value,
        np.maximum(total, TINY),
        out=np.zeros_like(value),
        where=total > 0.0,
    )


def link_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(left * left, axis=-1, dtype=np.float64)
        * np.sum(right * right, axis=-1, dtype=np.float64)
    )
    return (numerator / np.where(denominator == 0.0, 1e-12, denominator)).astype(
        np.float32
    )


def spectrum_summary(base: np.ndarray, raw: np.ndarray) -> np.ndarray:
    base = normalize(base)
    raw = normalize(raw)
    bins = np.arange(192, dtype=np.float32)
    base_peak = np.argmax(base, axis=-1).astype(np.float32) / 191.0
    raw_peak = np.argmax(raw, axis=-1).astype(np.float32) / 191.0
    base_mean = np.sum(base * bins, axis=-1) / 191.0
    raw_mean = np.sum(raw * bins, axis=-1) / 191.0
    base_var = np.sum(
        base * (bins[None, None] / 191.0 - base_mean[..., None]) ** 2, axis=-1
    )
    raw_var = np.sum(
        raw * (bins[None, None] / 191.0 - raw_mean[..., None]) ** 2, axis=-1
    )
    base_entropy = -np.sum(
        base * np.log(np.maximum(base, 1e-12)), axis=-1
    ) / np.log(192.0)
    raw_entropy = -np.sum(
        raw * np.log(np.maximum(raw, 1e-12)), axis=-1
    ) / np.log(192.0)
    return np.stack(
        (
            link_cosine(base, raw),
            np.sum(np.abs(base - raw), axis=-1),
            base_peak,
            raw_peak,
            np.abs(base_peak - raw_peak),
            base_mean,
            raw_mean,
            np.sqrt(np.maximum(base_var, 0.0)),
            np.sqrt(np.maximum(raw_var, 0.0)),
            base_entropy,
            raw_entropy,
            np.max(base, axis=-1),
            np.max(raw, axis=-1),
        ),
        axis=-1,
    ).astype(np.float32)


def build_cache(args: argparse.Namespace) -> None:
    paths = (FEATURES, BEST_ALPHA, BEST_GAIN, SCORE_GRID)
    if all(path.is_file() for path in paths) and not args.force_features:
        return
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    rows = np.load(PATH_CACHE / "rows.npy").astype(np.int64)
    global_ids = query_ids[rows]
    base = np.load(E65_DIR / "v9_base_pdp_f16.npy", mmap_mode="r")
    raw = np.load(E65_DIR / "raw_path_pdp_f16.npy", mmap_mode="r")
    pretrained = np.load(V9_ALPHA, mmap_mode="r")
    context = np.load(POINT_CONTEXT, mmap_mode="r")
    point_extra = np.asarray(context[global_ids, 120:], dtype=np.float32)
    link = link_coordinates().astype(np.float32)
    truth_h = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth_blocks = []
    for start in range(0, len(global_ids), args.batch_size):
        end = min(start + args.batch_size, len(global_ids))
        channel = torch.as_tensor(
            np.asarray(truth_h[global_ids[start:end]], dtype=np.complex64),
            device=args.device,
        )
        with torch.inference_mode():
            _, truth_pdp = truth_views(channel)
        truth_blocks.append(truth_pdp.cpu().numpy().astype(np.float32))
    truth = np.concatenate(truth_blocks).reshape(len(global_ids), 1024, 192)
    feature_out = np.lib.format.open_memmap(
        FEATURES,
        mode="w+",
        dtype=np.float16,
        shape=(len(global_ids), 1024, FEATURE_COUNT),
    )
    alpha_out = np.lib.format.open_memmap(
        BEST_ALPHA, mode="w+", dtype=np.float16, shape=(len(global_ids), 1024)
    )
    gain_out = np.lib.format.open_memmap(
        BEST_GAIN, mode="w+", dtype=np.float16, shape=(len(global_ids), 1024)
    )
    score_out = np.lib.format.open_memmap(
        SCORE_GRID,
        mode="w+",
        dtype=np.float16,
        shape=(len(global_ids), 1024, len(ALPHA_GRID)),
    )
    for start in range(0, len(global_ids), args.batch_size):
        end = min(start + args.batch_size, len(global_ids))
        local_base = np.asarray(base[start:end], dtype=np.float32).reshape(
            end - start, 1024, 192
        )
        local_raw = np.asarray(raw[start:end], dtype=np.float32).reshape(
            end - start, 1024, 192
        )
        summary = spectrum_summary(local_base, local_raw)
        point = np.broadcast_to(
            point_extra[start:end, None, :],
            (end - start, 1024, point_extra.shape[1]),
        )
        link_block = np.broadcast_to(link[None], (end - start, 1024, link.shape[1]))
        feature = np.concatenate(
            (
                np.asarray(pretrained[start:end], dtype=np.float32)[..., None],
                summary,
                point,
                link_block,
            ),
            axis=-1,
        )
        if feature.shape[-1] != FEATURE_COUNT:
            raise RuntimeError(f"unexpected feature shape {feature.shape}")
        score = np.empty(
            (end - start, 1024, len(ALPHA_GRID)), dtype=np.float32
        )
        for index, alpha in enumerate(ALPHA_GRID):
            candidate = normalize((1.0 - alpha) * local_base + alpha * local_raw)
            score[..., index] = link_cosine(candidate, truth[start:end])
        best_index = np.argmax(score, axis=-1)
        best_score = np.take_along_axis(score, best_index[..., None], axis=-1)[..., 0]
        feature_out[start:end] = feature.astype(np.float16)
        alpha_out[start:end] = ALPHA_GRID[best_index].astype(np.float16)
        gain_out[start:end] = (best_score - score[..., 0]).astype(np.float16)
        score_out[start:end] = score.astype(np.float16)
    feature_out.flush()
    alpha_out.flush()
    gain_out.flush()
    score_out.flush()


def fit_model(
    features: np.ndarray, target: np.ndarray, args: argparse.Namespace, seed: int
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=args.iterations,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=seed,
    ).fit(features, target)


def interpolated_score(score_grid: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    upper = np.searchsorted(ALPHA_GRID, alpha, side="right")
    upper = np.clip(upper, 1, len(ALPHA_GRID) - 1)
    lower = upper - 1
    left = ALPHA_GRID[lower]
    right = ALPHA_GRID[upper]
    weight = (alpha - left) / np.maximum(right - left, 1e-8)
    low = np.take_along_axis(score_grid, lower[..., None], axis=-1)[..., 0]
    high = np.take_along_axis(score_grid, upper[..., None], axis=-1)[..., 0]
    return low * (1.0 - weight) + high * weight


def train_oof(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata = np.load(OOF_META, allow_pickle=False)
    rows = np.load(PATH_CACHE / "rows.npy").astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)[rows]
    features = np.load(FEATURES, mmap_mode="r")
    best_alpha = np.load(BEST_ALPHA, mmap_mode="r")
    best_gain = np.load(BEST_GAIN, mmap_mode="r")
    flat_features = features.reshape(-1, FEATURE_COUNT)
    flat_alpha = best_alpha.reshape(-1)
    flat_gain = best_gain.reshape(-1)
    predicted_alpha = np.zeros(best_alpha.shape, dtype=np.float32)
    predicted_gain = np.zeros(best_gain.shape, dtype=np.float32)
    rng = np.random.default_rng(args.random_state)
    for fold in range(5):
        train_points = np.flatnonzero(fold_id != fold)
        held_points = np.flatnonzero(fold_id == fold)
        train_rows = (
            train_points[:, None] * 1024 + np.arange(1024, dtype=np.int64)[None]
        ).reshape(-1)
        sample = rng.choice(
            train_rows, size=min(args.train_samples, len(train_rows)), replace=False
        )
        alpha_model = fit_model(
            np.asarray(flat_features[sample], dtype=np.float32),
            np.asarray(flat_alpha[sample], dtype=np.float32),
            args,
            args.random_state + fold,
        )
        gain_model = fit_model(
            np.asarray(flat_features[sample], dtype=np.float32),
            np.asarray(flat_gain[sample], dtype=np.float32),
            args,
            args.random_state + 100 + fold,
        )
        for start in range(0, len(held_points), 32):
            ids = held_points[start : start + 32]
            local = np.asarray(features[ids], dtype=np.float32).reshape(
                -1, FEATURE_COUNT
            )
            predicted_alpha[ids] = alpha_model.predict(local).reshape(len(ids), 1024)
            predicted_gain[ids] = gain_model.predict(local).reshape(len(ids), 1024)
        joblib.dump(alpha_model, ARTIFACT_DIR / f"fold{fold}_alpha.joblib", compress=3)
        joblib.dump(gain_model, ARTIFACT_DIR / f"fold{fold}_gain.joblib", compress=3)
        print(f"V9 path action calibrator fold {fold + 1}/5", flush=True)
    np.savez_compressed(
        OOF_PREDICTION,
        predicted_alpha=predicted_alpha.astype(np.float16),
        predicted_gain=predicted_gain.astype(np.float16),
        fold_id=fold_id,
    )
    return predicted_alpha, predicted_gain, fold_id


def candidate_actions(
    predicted_alpha: np.ndarray,
    predicted_gain: np.ndarray,
    fold_id: np.ndarray,
    score_grid: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for quantile in (0.0, 0.20, 0.35, 0.50, 0.60, 0.70, 0.80, 0.90):
        active = np.zeros(predicted_gain.shape, dtype=bool)
        for fold in range(5):
            mask = fold_id == fold
            threshold = float(np.quantile(predicted_gain[mask], quantile))
            active[mask] = predicted_gain[mask] >= threshold
        for scale in (0.50, 0.75, 1.00, 1.25, 1.50):
            alpha = np.where(
                active, np.clip(scale * predicted_alpha, 0.0, 1.0), 0.0
            )
            selected = interpolated_score(score_grid, alpha)
            baseline = np.asarray(score_grid[..., 0], dtype=np.float32)
            fold_delta = [
                float(np.mean(selected[fold_id == fold] - baseline[fold_id == fold]))
                for fold in range(5)
            ]
            rows.append(
                {
                    "gain_quantile": quantile,
                    "alpha_scale": scale,
                    "active_fraction": float(np.mean(active)),
                    "link_cosine": float(np.mean(selected)),
                    "link_delta": float(np.mean(selected - baseline)),
                    "positive_folds": int(sum(value > 0.0 for value in fold_delta)),
                    "fold_link_delta": fold_delta,
                    "alpha": alpha.astype(np.float16),
                }
            )
    rows.sort(key=lambda row: float(row["link_delta"]), reverse=True)
    return rows


def evaluate_full_channel(
    rows: list[dict[str, object]], fold_id: np.ndarray, device: torch.device
) -> tuple[list[dict[str, object]], dict[str, object]]:
    base_channel = np.load(E65_DIR / "v9_base_channel.npy", mmap_mode="r")
    base_pdp = np.load(E65_DIR / "v9_base_pdp_f16.npy", mmap_mode="r")
    raw_pdp = np.load(E65_DIR / "raw_path_pdp_f16.npy", mmap_mode="r")
    metadata = np.load(OOF_META, allow_pickle=False)
    path_rows = np.load(PATH_CACHE / "rows.npy").astype(np.int64)
    query_ids = metadata["query_ids"].astype(np.int64)[path_rows]
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth_channel = np.asarray(truth_all[query_ids], dtype=np.complex64)
    base_tensor = torch.as_tensor(np.array(base_channel, copy=True), device=device)
    truth_tensor = torch.as_tensor(truth_channel, device=device)
    truth_pas, truth_pdp = truth_views(truth_tensor)
    base_pdp_tensor = torch.as_tensor(
        np.asarray(base_pdp, dtype=np.float32), device=device
    )
    raw_pdp_tensor = torch.as_tensor(
        np.asarray(raw_pdp, dtype=np.float32), device=device
    )
    baseline_values = point_metrics(base_tensor, truth_tensor, truth_pas, truth_pdp)
    baseline = statistics(*baseline_values)
    evaluated = []
    seen: set[tuple[float, float]] = set()
    for row in rows[:16]:
        key = (float(row["gain_quantile"]), float(row["alpha_scale"]))
        if key in seen:
            continue
        seen.add(key)
        alpha = torch.as_tensor(
            np.asarray(row["alpha"], dtype=np.float32), device=device
        ).reshape(len(base_tensor), 256, 4)
        desired = (
            (1.0 - alpha[..., None]) * base_pdp_tensor
            + alpha[..., None] * raw_pdp_tensor
        )
        desired = desired / desired.sum(dim=3, keepdim=True).clamp_min(TINY)
        for cycles in (1, 2, 4):
            with torch.inference_mode():
                prediction = project(base_tensor, desired, cycles, "pas_pdp")
                values = point_metrics(prediction, truth_tensor, truth_pas, truth_pdp)
            current = statistics(*values)
            fold_rows = []
            for fold in range(5):
                mask = fold_id == fold
                local = statistics(*values, mask)
                anchor = statistics(*baseline_values, mask)
                fold_rows.append(
                    {
                        "fold": fold,
                        "delta": float(local["score"] - anchor["score"]),
                    }
                )
            evaluated.append(
                {
                    "candidate": (
                        f"q{int(round(100 * key[0])):02d}_"
                        f"s{int(round(100 * key[1])):03d}_c{cycles}"
                    ),
                    "gain_quantile": key[0],
                    "alpha_scale": key[1],
                    "cycles": cycles,
                    "active_fraction": float(row["active_fraction"]),
                    "link_delta": float(row["link_delta"]),
                    **current,
                    "folds": fold_rows,
                    "positive_folds": int(
                        sum(local["delta"] > 0.0 for local in fold_rows)
                    ),
                }
            )
    evaluated.sort(key=lambda row: float(row["score"]), reverse=True)
    return evaluated, baseline


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (
        OOF_META,
        PATH_CACHE,
        E65_DIR / "v9_base_pdp_f16.npy",
        E65_DIR / "raw_path_pdp_f16.npy",
        E65_DIR / "v9_base_channel.npy",
        POINT_CONTEXT,
        V9_ALPHA,
    )
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    build_cache(args)
    predicted_alpha, predicted_gain, fold_id = train_oof(args)
    score_grid = np.asarray(np.load(SCORE_GRID, mmap_mode="r"), dtype=np.float32)
    link_rows = candidate_actions(
        predicted_alpha, predicted_gain, fold_id, score_grid
    )
    evaluated, baseline = evaluate_full_channel(
        link_rows, fold_id, torch.device(args.device)
    )
    best_alpha = np.asarray(np.load(BEST_ALPHA, mmap_mode="r"), dtype=np.float32)
    best_gain = np.asarray(np.load(BEST_GAIN, mmap_mode="r"), dtype=np.float32)
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "configuration": vars(args),
        "baseline": baseline,
        "oracle": {
            "link_baseline_cosine": float(np.mean(score_grid[..., 0])),
            "link_oracle_cosine": float(np.mean(np.max(score_grid, axis=-1))),
            "link_oracle_gain": float(
                np.mean(np.max(score_grid, axis=-1) - score_grid[..., 0])
            ),
            "fraction_no_action": float(np.mean(best_alpha == 0.0)),
            "mean_best_gain": float(np.mean(best_gain)),
        },
        "selected": evaluated[0],
        "top_full_channel": evaluated[:20],
        "top_link_actions": [
            {key: value for key, value in row.items() if key != "alpha"}
            for row in link_rows[:20]
        ],
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
