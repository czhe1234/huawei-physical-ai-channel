#!/usr/bin/env python3
"""Realize the v480 inferred PAS/PDP targets with exact alternating projections."""

from __future__ import annotations

import json

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v479_online_spectral_target_inversion as v479
import analyze_v480_nested_online_pas_pdp_inversion as v480
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
import generate_v427_sionna_conventional_pas_dual_lock_diagnostic as v427
import train_v469_neural_sionna_calibrator as v469
from project_paths import PATHS


VERSION_NUMBER = 481
VERSION = "v481_exact_dual_spectral_projection"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = CACHE / "v481_Round1_Test_Channel.npy"
CHECKPOINT_CYCLES = (1, 2, 5, 10, 20, 40)


def exact_pas_projection(channel: np.ndarray, desired: np.ndarray) -> np.ndarray:
    spectrum = v427.array_forward(channel)
    magnitude = np.abs(spectrum).astype(np.float32)
    phase = spectrum / np.maximum(magnitude, 1e-30)
    phase = np.where(magnitude > 1e-20, phase, np.complex64(1.0 + 0.0j))
    total = np.sum(magnitude * magnitude, axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    desired_magnitude = np.sqrt(
        np.maximum(desired * total[:, None, None, None, :, :], 0.0)
    ).astype(np.float32)
    return v427.array_inverse(desired_magnitude * phase)


def generate_and_measure(
    arrays: list[np.ndarray],
    alpha: np.ndarray,
    intercept: float,
    pas_weight: float,
    batch_size: int = 8,
) -> tuple[list[dict[str, object]], int]:
    anchor = np.load(v469.OUTPUT, mmap_mode="r")
    pas_target = np.load(v480.PAS_TARGET, mmap_mode="r")
    pdp_target = np.load(v480.PDP_TARGET, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=anchor.shape
    )
    pas_dot = np.zeros((len(CHECKPOINT_CYCLES), len(arrays)), dtype=np.float64)
    pdp_dot = np.zeros_like(pas_dot)
    pas_vectors = 0
    pdp_vectors = 0
    target_pas_cosine = np.zeros(len(CHECKPOINT_CYCLES), dtype=np.float64)
    target_pdp_cosine = np.zeros(len(CHECKPOINT_CYCLES), dtype=np.float64)
    for start in range(0, 500, batch_size):
        end = min(start + batch_size, 500)
        base = np.asarray(anchor[start:end], dtype=np.complex64)
        desired_pas = np.asarray(pas_target[start:end], dtype=np.float32)
        desired_pdp = np.asarray(pdp_target[start:end], dtype=np.float32)
        desired_pas_l2 = v479.normalize_l2(
            desired_pas.transpose(0, 4, 5, 1, 2, 3).reshape(end - start, 4, 192, 256),
            -1,
        )
        desired_pdp_l2 = v479.normalize_l2(
            desired_pdp.reshape(end - start, 1024, 192), -1
        )
        source_pas = []
        source_pdp = []
        for value in arrays:
            pas, pdp, _ = v479.metric_features(
                np.asarray(value[start:end], dtype=np.complex64)
            )
            source_pas.append(pas)
            source_pdp.append(pdp)
        source_pas = np.stack(source_pas, axis=0)
        source_pdp = np.stack(source_pdp, axis=0)
        candidate = base.copy()
        checkpoint_id = 0
        for cycle in range(1, CHECKPOINT_CYCLES[-1] + 1):
            candidate = exact_pas_projection(candidate, desired_pas)
            candidate = v426.restore_desired_pdp(candidate, base, desired_pdp)
            if cycle != CHECKPOINT_CYCLES[checkpoint_id]:
                continue
            actual_pas, actual_pdp, _ = v479.metric_features(candidate)
            pas_dot[checkpoint_id] += np.einsum(
                "bqfd,kbqfd->k", actual_pas, source_pas, optimize=True
            )
            pdp_dot[checkpoint_id] += np.einsum(
                "bld,kbld->k", actual_pdp, source_pdp, optimize=True
            )
            target_pas_cosine[checkpoint_id] += float(
                np.sum(actual_pas * desired_pas_l2, dtype=np.float64)
            )
            target_pdp_cosine[checkpoint_id] += float(
                np.sum(actual_pdp * desired_pdp_l2, dtype=np.float64)
            )
            checkpoint_id += 1
            if checkpoint_id == len(CHECKPOINT_CYCLES):
                break
        output[start:end] = candidate
        pas_vectors += int(np.prod(desired_pas_l2.shape[:-1]))
        pdp_vectors += int(np.prod(desired_pdp_l2.shape[:-1]))
        if end % 50 == 0:
            output.flush()
            print(f"v481 exact projection {end}/500", flush=True)
    output.flush()
    rows = []
    for index, cycle in enumerate(CHECKPOINT_CYCLES):
        pas_similarity = pas_dot[index] / pas_vectors
        pdp_similarity = pdp_dot[index] / pdp_vectors
        kernel_similarity = pas_weight * pas_similarity + (1.0 - pas_weight) * pdp_similarity
        rows.append(
            {
                "cycles": cycle,
                "predicted_online_score": float(kernel_similarity @ alpha + intercept),
                "pas_target_cosine": float(target_pas_cosine[index] / pas_vectors),
                "pdp_target_cosine": float(target_pdp_cosine[index] / pdp_vectors),
                "pas_similarity_to_versions": pas_similarity.tolist(),
                "pdp_similarity_to_versions": pdp_similarity.tolist(),
            }
        )
    best_id = int(np.argmax([row["predicted_online_score"] for row in rows]))
    return rows, best_id


def regenerate(cycles: int, batch_size: int = 8) -> None:
    anchor = np.load(v469.OUTPUT, mmap_mode="r")
    pas_target = np.load(v480.PAS_TARGET, mmap_mode="r")
    pdp_target = np.load(v480.PDP_TARGET, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=anchor.shape
    )
    for start in range(0, 500, batch_size):
        end = min(start + batch_size, 500)
        base = np.asarray(anchor[start:end], dtype=np.complex64)
        desired_pas = np.asarray(pas_target[start:end], dtype=np.float32)
        desired_pdp = np.asarray(pdp_target[start:end], dtype=np.float32)
        candidate = base.copy()
        for _ in range(cycles):
            candidate = exact_pas_projection(candidate, desired_pas)
            candidate = v426.restore_desired_pdp(candidate, base, desired_pdp)
        output[start:end] = candidate
    output.flush()


def main() -> None:
    source = json.loads(v480.REPORT.read_text(encoding="utf-8"))
    selected_kernel = source["inherited_full_kernel"]
    pas_weight = float(selected_kernel["pas_weight"])
    alpha = np.asarray(source["full_fit"]["alpha"], dtype=np.float64)
    intercept = float(source["full_fit"]["intercept"])
    arrays = [np.load(path, mmap_mode="r") for _, _, path in v479.VERSIONS]
    CACHE.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows, best_id = generate_and_measure(arrays, alpha, intercept, pas_weight)
    selected_cycles = int(rows[best_id]["cycles"])
    if selected_cycles != CHECKPOINT_CYCLES[-1]:
        regenerate(selected_cycles)
    replay = v208.score_prediction(OUTPUT)
    baseline_replay = v208.score_prediction(v469.OUTPUT)
    selected_prediction = float(rows[best_id]["predicted_online_score"])
    passed = selected_prediction >= float(source["selected"]["realized_output_predicted_online_score"])
    report = {
        "method": VERSION,
        "status": "candidate_signal_passed" if passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Exact Gerchberg-Saxton-style alternating magnitude projections reduce the "
            "implementation gap between the v480 inferred PAS/PDP targets and realized H."
        ),
        "source": {
            "v480_realized_prediction": source["selected"]["realized_output_predicted_online_score"],
            "v480_ideal_prediction": source["selected"]["ideal_target_predicted_online_score"],
        },
        "cycle_search": rows,
        "selected_cycles": selected_cycles,
        "selected_predicted_online_score": selected_prediction,
        "replay": replay,
        "baseline_v469_replay": baseline_replay,
        "replay_delta_score": float(
            replay["score_traceback_leaked"] - baseline_replay["score_traceback_leaked"]
        ),
        "decision": passed,
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "cycle_search": rows,
        "selected_cycles": selected_cycles,
        "selected_predicted_online_score": selected_prediction,
        "replay_delta_score": report["replay_delta_score"],
        "passed": passed,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
