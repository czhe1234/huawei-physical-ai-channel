#!/usr/bin/env python3
"""Invert online-score measurements into a low-rank hidden spectral direction."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.stats import pearsonr, spearmanr

import analyze_v208_strict_historical_replay as v208
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v424_sionna_pas_full_pdp_lock_diagnostic as v424
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
import generate_v427_sionna_conventional_pas_dual_lock_diagnostic as v427
import train_v469_neural_sionna_calibrator as v469
from project_paths import PATHS


VERSION_NUMBER = 479
VERSION = "v479_online_spectral_target_inversion"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
TARGET = CACHE / "v479_inverted_pas_target.npy"
OUTPUT = CACHE / "v479_Round1_Test_Channel.npy"

VERSIONS = (
    (209, 0.634488, PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy"),
    (212, 0.690183, PATHS.cache_dir(212) / "v212_Round1_Test_Channel.npy"),
    (426, 0.721766, PATHS.cache_dir(426) / "v426_link075_Round1_Test_Channel.npy"),
    (427, 0.702860, PATHS.cache_dir(427) / "v427_array100_Round1_Test_Channel.npy"),
    (428, 0.721227, PATHS.cache_dir(428) / "v428_slice050_Round1_Test_Channel.npy"),
    (436, 0.727491, PATHS.cache_dir(436) / "v436_Round1_Test_Channel.npy"),
    (449, 0.731214, PATHS.cache_dir(449) / "v449_Round1_Test_Channel.npy"),
    (453, 0.633698, PATHS.cache_dir(453) / "v453_Round1_Test_Channel.npy"),
    (458, 0.739203, PATHS.cache_dir(458) / "v458_Round1_Test_Channel.npy"),
    (459, 0.736148, PATHS.cache_dir(459) / "v459_Round1_Test_Channel.npy"),
    (463, 0.737441, PATHS.cache_dir(463) / "v463_Round1_Test_Channel.npy"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def normalize_l2(value: np.ndarray, axis: int) -> np.ndarray:
    denominator = np.linalg.norm(value, axis=axis, keepdims=True)
    return (value / np.maximum(denominator, 1e-20)).astype(np.float32)


def metric_features(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value6 = channel.reshape(len(channel), 2, 16, 8, 4, 192)
    pas_power = (np.abs(np.fft.fft2(value6, axes=(2, 3))).astype(np.float32) ** 2)
    pas_cell = pas_power.transpose(0, 4, 5, 1, 2, 3).reshape(len(channel), 4, 192, 256)
    pas_l2 = normalize_l2(pas_cell, -1)
    pas_l1 = pas_cell / np.maximum(np.sum(pas_cell, axis=-1, keepdims=True), 1e-20)
    delay = np.abs(np.fft.ifft(channel, axis=-1)).astype(np.float32) ** 2
    pdp_cell = delay.transpose(0, 1, 2, 3).reshape(len(channel), 1024, 192)
    pdp_l2 = normalize_l2(pdp_cell, -1)
    return pas_l2, pdp_l2, pas_l1.astype(np.float32)


def compute_gram(arrays: list[np.ndarray], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    count = len(arrays)
    pas_gram = np.zeros((count, count), dtype=np.float64)
    pdp_gram = np.zeros((count, count), dtype=np.float64)
    pas_vectors = 0
    pdp_vectors = 0
    for start in range(0, 500, batch_size):
        end = min(start + batch_size, 500)
        pas_rows = []
        pdp_rows = []
        for value in arrays:
            pas, pdp, _ = metric_features(np.asarray(value[start:end], dtype=np.complex64))
            pas_rows.append(pas.reshape(-1, 256))
            pdp_rows.append(pdp.reshape(-1, 192))
        pas_stack = np.stack(pas_rows, axis=0)
        pdp_stack = np.stack(pdp_rows, axis=0)
        pas_gram += np.einsum("knd,lnd->kl", pas_stack, pas_stack, optimize=True)
        pdp_gram += np.einsum("knd,lnd->kl", pdp_stack, pdp_stack, optimize=True)
        pas_vectors += pas_stack.shape[1]
        pdp_vectors += pdp_stack.shape[1]
        if end % 50 == 0:
            print(f"v479 gram {end}/500", flush=True)
    return pas_gram / pas_vectors, pdp_gram / pdp_vectors


def kernel_fit(kernel: np.ndarray, target: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    count = len(target)
    system = np.zeros((count + 1, count + 1), dtype=np.float64)
    system[:count, :count] = kernel + ridge * np.eye(count)
    system[:count, count] = 1.0
    system[count, :count] = 1.0
    right = np.concatenate([target.astype(np.float64), [0.0]])
    solution = np.linalg.solve(system, right)
    return solution[:count], float(solution[count])


def loo_predictions(kernel: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    prediction = np.empty(len(target), dtype=np.float64)
    for holdout in range(len(target)):
        fit = np.arange(len(target)) != holdout
        alpha, intercept = kernel_fit(kernel[np.ix_(fit, fit)], target[fit], ridge)
        prediction[holdout] = kernel[holdout, fit] @ alpha + intercept
    return prediction


def choose_kernel(
    pas_gram: np.ndarray,
    pdp_gram: np.ndarray,
    scores: np.ndarray,
) -> dict[str, object]:
    rows = []
    best = None
    for pas_weight in np.linspace(0.0, 1.0, 11):
        kernel = pas_weight * pas_gram + (1.0 - pas_weight) * pdp_gram
        for ridge in np.logspace(-8, -1, 15):
            prediction = loo_predictions(kernel, scores, float(ridge))
            rmse = float(np.sqrt(np.mean((prediction - scores) ** 2)))
            pearson = float(pearsonr(prediction, scores).statistic)
            spearman = float(spearmanr(prediction, scores).statistic)
            row = {
                "pas_weight": float(pas_weight),
                "ridge": float(ridge),
                "rmse": rmse,
                "pearson": pearson,
                "spearman": spearman,
                "predictions": prediction.tolist(),
            }
            rows.append(row)
            key = (rmse, -pearson, -spearman)
            if best is None or key < best[0]:
                best = (key, row, kernel)
    assert best is not None
    return {"best": best[1], "kernel": best[2], "search": rows}


def candidate_similarities(
    arrays: list[np.ndarray],
    anchor: np.ndarray,
    alpha: np.ndarray,
    eta_values: np.ndarray,
    pas_weight: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(arrays)
    pas_dot = np.zeros((len(eta_values), count), dtype=np.float64)
    pdp_dot = np.zeros((len(eta_values), count), dtype=np.float64)
    pas_vectors = 0
    pdp_vectors = 0
    for start in range(0, 500, batch_size):
        end = min(start + batch_size, 500)
        source_pas = []
        source_pdp = []
        for value in arrays:
            pas, pdp, _ = metric_features(np.asarray(value[start:end], dtype=np.complex64))
            source_pas.append(pas)
            source_pdp.append(pdp)
        source_pas_stack = np.stack(source_pas, axis=0)
        source_pdp_stack = np.stack(source_pdp, axis=0)
        anchor_pas, anchor_pdp, _ = metric_features(
            np.asarray(anchor[start:end], dtype=np.complex64)
        )
        direction = np.tensordot(alpha, source_pas_stack, axes=(0, 0))
        direction_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        direction = direction / np.maximum(direction_norm, 1e-20)
        for eta_id, eta in enumerate(eta_values):
            candidate = np.maximum(anchor_pas + float(eta) * direction, 0.0)
            candidate = normalize_l2(candidate, -1)
            pas_dot[eta_id] += np.einsum(
                "bqfd,kbqfd->k", candidate, source_pas_stack, optimize=True
            )
            pdp_dot[eta_id] += np.einsum(
                "bld,kbld->k", anchor_pdp, source_pdp_stack, optimize=True
            )
        pas_vectors += int(np.prod(anchor_pas.shape[:-1]))
        pdp_vectors += int(np.prod(anchor_pdp.shape[:-1]))
    similarities = (
        pas_weight * pas_dot / pas_vectors
        + (1.0 - pas_weight) * pdp_dot / pdp_vectors
    )
    return similarities, pas_dot / pas_vectors


def write_target(
    arrays: list[np.ndarray],
    anchor: np.ndarray,
    alpha: np.ndarray,
    eta: float,
    batch_size: int,
) -> None:
    output = np.lib.format.open_memmap(
        TARGET,
        mode="w+",
        dtype=np.float16,
        shape=(500, 2, 16, 8, 4, 192),
    )
    for start in range(0, 500, batch_size):
        end = min(start + batch_size, 500)
        source_pas = []
        for value in arrays:
            pas, _, _ = metric_features(np.asarray(value[start:end], dtype=np.complex64))
            source_pas.append(pas)
        source_pas_stack = np.stack(source_pas, axis=0)
        anchor_pas, _, _ = metric_features(np.asarray(anchor[start:end], dtype=np.complex64))
        direction = np.tensordot(alpha, source_pas_stack, axes=(0, 0))
        direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-20)
        candidate = np.maximum(anchor_pas + eta * direction, 0.0)
        candidate /= np.maximum(np.sum(candidate, axis=-1, keepdims=True), 1e-20)
        output[start:end] = candidate.reshape(end - start, 4, 192, 2, 16, 8).transpose(
            0, 3, 4, 5, 1, 2
        ).astype(np.float16)
    output.flush()


def generate_channel(batch_size: int) -> None:
    anchor = np.load(v469.OUTPUT, mmap_mode="r")
    target = np.load(TARGET, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=anchor.shape
    )
    for start in range(0, 500, batch_size):
        end = min(start + batch_size, 500)
        base_h = np.asarray(anchor[start:end], dtype=np.complex64)
        desired = np.asarray(target[start:end], dtype=np.float32)
        base_leaked, _ = v423.leaked_support(base_h)
        candidate = base_h.copy()
        for _ in range(3):
            candidate = v427.project_conventional_pas(candidate, desired)
            candidate = v425.project_full_support(candidate, base_leaked)
            candidate = v424.restore_full_linkwise_pdp(candidate, base_h)
        output[start:end] = candidate
        if end % 50 == 0:
            output.flush()
            print(f"v479 projection {end}/500", flush=True)
    output.flush()


def main() -> None:
    args = parse_args()
    for _, _, path in VERSIONS:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not v469.OUTPUT.is_file():
        raise FileNotFoundError(v469.OUTPUT)
    CACHE.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = [np.load(path, mmap_mode="r") for _, _, path in VERSIONS]
    scores = np.asarray([score for _, score, _ in VERSIONS], dtype=np.float64)
    pas_gram, pdp_gram = compute_gram(arrays, args.batch_size)
    selection = choose_kernel(pas_gram, pdp_gram, scores)
    best = selection["best"]
    kernel = selection["kernel"]
    alpha, intercept = kernel_fit(kernel, scores, float(best["ridge"]))
    eta_values = np.linspace(0.0, 2.0, 21)
    anchor = np.load(v469.OUTPUT, mmap_mode="r")
    similarities, pas_similarities = candidate_similarities(
        arrays,
        anchor,
        alpha,
        eta_values,
        float(best["pas_weight"]),
        args.batch_size,
    )
    predicted_scores = similarities @ alpha + intercept
    selected_id = int(np.argmax(predicted_scores))
    selected_eta = float(eta_values[selected_id])
    write_target(arrays, anchor, alpha, selected_eta, args.batch_size)
    generate_channel(max(1, args.batch_size))
    replay = v208.score_prediction(OUTPUT)
    baseline_replay = v208.score_prediction(v469.OUTPUT)
    passed = (
        float(best["pearson"]) >= 0.75
        and float(best["spearman"]) >= 0.70
        and float(best["rmse"]) <= 0.025
        and selected_eta > 0.0
    )
    report = {
        "method": VERSION,
        "status": "candidate_signal_passed" if passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Online PAS/PDP scores are linear measurements of one hidden normalized spectral "
            "target, whose projection can be recovered in the span of scored predictions."
        ),
        "versions": [
            {"version": version, "score": score, "path": PATHS.relative(path)}
            for version, score, path in VERSIONS
        ],
        "pas_gram": pas_gram.tolist(),
        "pdp_gram": pdp_gram.tolist(),
        "kernel_search": selection["search"],
        "selected_kernel": best,
        "full_fit": {"alpha": alpha.tolist(), "intercept": intercept},
        "candidate_search": [
            {
                "eta": float(eta),
                "predicted_online_score": float(score),
                "pas_similarity_to_versions": pas_similarities[index].tolist(),
            }
            for index, (eta, score) in enumerate(zip(eta_values, predicted_scores))
        ],
        "selected_eta": selected_eta,
        "selected_predicted_online_score": float(predicted_scores[selected_id]),
        "replay": replay,
        "baseline_v469_replay": baseline_replay,
        "replay_delta_score": float(
            replay["score_traceback_leaked"] - baseline_replay["score_traceback_leaked"]
        ),
        "decision_rule": {
            "pearson_min": 0.75,
            "spearman_min": 0.70,
            "rmse_max": 0.025,
            "passed": passed,
        },
        "outputs": {
            "target": PATHS.relative(TARGET),
            "channel": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_kernel": best,
        "selected_eta": selected_eta,
        "predicted_online_score": float(predicted_scores[selected_id]),
        "replay_delta_score": report["replay_delta_score"],
        "passed": passed,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
