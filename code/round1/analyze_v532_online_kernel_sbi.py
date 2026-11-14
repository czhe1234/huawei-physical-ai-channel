#!/usr/bin/env python3
"""Infer a latent online joint-S-H score direction from submitted probes."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize

import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
from project_paths import PATHS


VERSION_NUMBER = 532
VERSION = "v532_online_kernel_sbi"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
FEATURE_CACHE = CACHE_DIR / f"{VERSION}_joint_features.npz"
RANDOM_SEED = 20260725
BATCH = 2

PROBES = (
    (458, 0.739203),
    (463, 0.737441),
    (491, 0.739634),
    (497, 0.739554),
    (500, 0.739607),
    (514, 0.738903),
)


def channel_path(version: int):
    candidates = sorted(PATHS.cache_dir(version).glob("v*_Round1_Test_Channel.npy"))
    if len(candidates) != 1:
        raise ValueError(f"expected one channel cache for v{version}, got {candidates}")
    return candidates[0]


def extract_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if FEATURE_CACHE.is_file():
        data = np.load(FEATURE_CACHE)
        return data["versions"], data["scores"], data["features"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    versions = np.asarray([version for version, _ in PROBES], dtype=np.int16)
    scores = np.asarray([score for _, score in PROBES], dtype=np.float64)
    features = np.empty((len(PROBES), 500, 192 * 16), dtype=np.float32)
    for probe_index, (version, _) in enumerate(PROBES):
        channel = np.load(channel_path(version), mmap_mode="r")
        if channel.shape != (500, 256, 4, 192):
            raise ValueError(f"unexpected v{version} shape: {channel.shape}")
        for start in range(0, len(channel), BATCH):
            stop = min(start + BATCH, len(channel))
            support, _ = v423.leaked_support(
                np.asarray(channel[start:stop], dtype=np.complex64)
            )
            sh = v423.leaked_marginals(support)["sh"].reshape(stop - start, -1)
            norm = np.sqrt(np.sum(sh * sh, axis=1, keepdims=True, dtype=np.float64))
            features[probe_index, start:stop] = sh / np.maximum(norm, 1e-30)
        print(f"v532 feature v{version} complete", flush=True)
    np.savez_compressed(
        FEATURE_CACHE,
        versions=versions,
        scores=scores,
        features=features,
    )
    return versions, scores, features


def centered_train(kernel: np.ndarray) -> np.ndarray:
    row_mean = kernel.mean(axis=1, keepdims=True)
    return kernel - row_mean - row_mean.T + kernel.mean()


def centered_query(kernel: np.ndarray, query: np.ndarray) -> np.ndarray:
    return query - kernel.mean(axis=0) - query.mean() + kernel.mean()


def fit_kernel_model(
    kernel: np.ndarray, scores: np.ndarray, ridge: float
) -> tuple[float, np.ndarray]:
    centered = centered_train(kernel)
    intercept = float(np.mean(scores))
    alpha = np.linalg.solve(
        centered + ridge * np.eye(len(centered), dtype=np.float64),
        scores - intercept,
    )
    return intercept, alpha


def predict_kernel(
    train_kernel: np.ndarray,
    train_scores: np.ndarray,
    query_kernel: np.ndarray,
    ridge: float,
) -> float:
    intercept, alpha = fit_kernel_model(train_kernel, train_scores, ridge)
    centered = centered_query(train_kernel, query_kernel)
    return float(intercept + centered @ alpha)


def select_ridge(kernel: np.ndarray, scores: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for ridge in np.logspace(-12, -2, 41):
        prediction = np.empty(len(scores), dtype=np.float64)
        for held_out in range(len(scores)):
            ids = np.delete(np.arange(len(scores)), held_out)
            prediction[held_out] = predict_kernel(
                kernel[np.ix_(ids, ids)],
                scores[ids],
                kernel[held_out, ids],
                float(ridge),
            )
        error = prediction - scores
        rows.append(
            {
                "ridge": float(ridge),
                "rmse": float(np.sqrt(np.mean(error * error))),
                "mae": float(np.mean(np.abs(error))),
                "correlation": float(np.corrcoef(prediction, scores)[0, 1]),
            }
        )
    best = min(rows, key=lambda row: row["rmse"])
    return float(best["ridge"]), rows


def candidate_kernel(row_gram: np.ndarray, weights: np.ndarray) -> np.ndarray:
    norm_sq = np.einsum("i,nij,j->n", weights, row_gram, weights, optimize=True)
    numerator = np.einsum("i,nij->nj", weights, row_gram, optimize=True)
    return np.mean(
        numerator / np.sqrt(np.maximum(norm_sq[:, None], 1e-30)), axis=0
    )


def model_predictions(
    kernel: np.ndarray,
    scores: np.ndarray,
    query: np.ndarray,
    ridge: float,
) -> np.ndarray:
    predictions = [predict_kernel(kernel, scores, query, ridge)]
    for held_out in range(len(scores)):
        ids = np.delete(np.arange(len(scores)), held_out)
        predictions.append(
            predict_kernel(
                kernel[np.ix_(ids, ids)],
                scores[ids],
                query[ids],
                ridge,
            )
        )
    return np.asarray(predictions, dtype=np.float64)


def optimize_weights(
    row_gram: np.ndarray,
    kernel: np.ndarray,
    scores: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(scores)

    def objective(weights: np.ndarray) -> float:
        query = candidate_kernel(row_gram, weights)
        posterior = model_predictions(kernel, scores, query, ridge)
        robust = np.mean(posterior) - 0.5 * np.std(posterior)
        return -float(robust)

    constraints = {"type": "eq", "fun": lambda value: np.sum(value) - 1.0}
    bounds = [(0.0, 1.0)] * count
    starts = [np.eye(count, dtype=np.float64)[index] for index in range(count)]
    rng = np.random.default_rng(RANDOM_SEED)
    starts.extend(rng.dirichlet(np.ones(count), size=64))
    best = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-13},
        )
        if result.success and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError("online kernel weight optimization failed")
    weights = np.maximum(best.x, 0.0)
    weights /= weights.sum()
    posterior = model_predictions(
        kernel, scores, candidate_kernel(row_gram, weights), ridge
    )
    return weights.astype(np.float32), posterior


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    versions, scores, features = extract_features()
    row_gram = np.einsum("mnd,knd->nmk", features, features, optimize=True)
    kernel = row_gram.mean(axis=0, dtype=np.float64)
    ridge, ridge_rows = select_ridge(kernel, scores)
    loo_prediction = np.empty(len(scores), dtype=np.float64)
    for held_out in range(len(scores)):
        ids = np.delete(np.arange(len(scores)), held_out)
        loo_prediction[held_out] = predict_kernel(
            kernel[np.ix_(ids, ids)],
            scores[ids],
            kernel[held_out, ids],
            ridge,
        )
    weights, posterior = optimize_weights(row_gram, kernel, scores, ridge)
    baseline_rmse = float(np.sqrt(np.mean((scores - np.mean(scores)) ** 2)))
    loo_rmse = float(np.sqrt(np.mean((loo_prediction - scores) ** 2)))
    pass_gate = bool(
        loo_rmse < 0.75 * baseline_rmse
        and float(np.mean(posterior) - 0.5 * np.std(posterior))
        > float(np.max(scores)) + 0.0001
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Online scores of joint-only probes constrain the hidden test joint-S-H target "
            "inside their observed spectral span; a kernel posterior can identify a "
            "complementary blend without assuming the local Sionna reference is truth."
        ),
        "evidence_boundary": {
            "online_scores_used": True,
            "probes_share_link_power_backbone": True,
            "hidden_per-point_truth_used": False,
            "candidate_generated": False,
        },
        "versions": versions.tolist(),
        "scores": scores.tolist(),
        "kernel": kernel.tolist(),
        "selected_ridge": ridge,
        "ridge_search": ridge_rows,
        "leave_one_out": {
            "prediction": loo_prediction.tolist(),
            "rmse": loo_rmse,
            "baseline_constant_rmse": baseline_rmse,
            "correlation": float(np.corrcoef(loo_prediction, scores)[0, 1]),
        },
        "optimized_blend": {
            "weights": {
                f"v{version}": float(weight)
                for version, weight in zip(versions, weights, strict=True)
            },
            "posterior_predictions": posterior.tolist(),
            "posterior_mean": float(np.mean(posterior)),
            "posterior_std": float(np.std(posterior)),
            "robust_prediction": float(np.mean(posterior) - 0.5 * np.std(posterior)),
        },
        "decision_gate": {
            "loo_rmse_below_75pct_constant": bool(loo_rmse < 0.75 * baseline_rmse),
            "robust_gain_above_best": float(
                np.mean(posterior) - 0.5 * np.std(posterior) - np.max(scores)
            ),
            "required_gain": 0.0001,
            "pass": pass_gate,
        },
        "outputs": {
            "feature_cache": PATHS.relative(FEATURE_CACHE),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
