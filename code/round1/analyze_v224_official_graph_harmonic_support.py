#!/usr/bin/env python3
"""v224 analysis: official graph-harmonic full support propagation.

v223 showed that full tensor official train-H support has weak positive signal,
but one-hop train->test KNN is not enough. This probe adds a strictly official
transductive ingredient: the combined train/test coordinate graph. Train nodes
hold official train-H support labels; test nodes are unlabeled and receive a
harmonic propagation target through test-test and train-test edges.

The probe is analysis-only. It is not a recommended candidate because the
propagation is deterministic rather than a neural Physical-AI model. If it
beats v223 by a meaningful margin, the next step is a graph neural field with
the same official data boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v223_official_fulltensor_support_signal as v223
import generate_v218_official_multispectrum_physics_ai as v218
from generate_v87_easy_field_projection import summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 224
VERSION = "v224_official_graph_harmonic_support"

OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v224_official_graph_harmonic_support.json"

GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

AVG_BATCH = 3
K_TRAIN = 56
K_TEST = 40
FLOOR = 1e-12


def normalize_full(x: np.ndarray) -> np.ndarray:
    return v223.normalize_full(x)


def effective_distance(
    query_pos: np.ndarray,
    query_blocked: np.ndarray,
    bank_pos: np.ndarray,
    bank_blocked: np.ndarray,
) -> np.ndarray:
    return v218.effective_distance(query_pos, query_blocked, bank_pos, bank_blocked)


def train_test_weights(
    train_pos: np.ndarray,
    train_blocked: np.ndarray,
    test_pos: np.ndarray,
    test_blocked: np.ndarray,
    valid_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx_train, w_train = v218.neighbor_index_weights(
        test_pos,
        test_blocked,
        train_pos[valid_ids],
        train_blocked[valid_ids],
        k=K_TRAIN,
    )

    dist_tt = effective_distance(test_pos, test_blocked, test_pos, test_blocked)
    np.fill_diagonal(dist_tt, np.inf)
    idx_test = np.argpartition(dist_tt, K_TEST - 1, axis=1)[:, :K_TEST]
    dsel = np.take_along_axis(dist_tt, idx_test, axis=1)
    order = np.argsort(dsel, axis=1)
    idx_test = np.take_along_axis(idx_test, order, axis=1).astype(np.int32)
    dsel = np.take_along_axis(dsel, order, axis=1)
    same_regime = regime_of(test_blocked)[:, None] == regime_of(test_blocked)[idx_test]
    w_test = np.exp(-dsel / 18.0).astype(np.float32) * np.where(same_regime, 1.22, 0.68).astype(np.float32)
    w_test /= np.maximum(w_test.sum(axis=1, keepdims=True), TINY)
    return idx_train.astype(np.int32), w_train.astype(np.float32), idx_test, w_test.astype(np.float32)


def harmonic_support(
    name: str,
    bank: np.ndarray,
    valid_ids: np.ndarray,
    idx_train: np.ndarray,
    w_train: np.ndarray,
    idx_test: np.ndarray,
    w_test: np.ndarray,
    init: np.ndarray,
    train_mix: float,
    iterations: int,
) -> np.ndarray:
    output_path = OUT_DIR / f"v224_{name}_mix{train_mix:.2f}_it{iterations}.npy"
    if output_path.is_file():
        return np.load(output_path, mmap_mode="r")
    current = np.asarray(init, dtype=np.float32).copy()
    train_shape = (1,) * (bank.ndim - 1)
    for iteration in range(1, iterations + 1):
        nxt = np.empty_like(current)
        for start in range(0, len(current), AVG_BATCH):
            end = min(start + AVG_BATCH, len(current))
            train_block = np.zeros((end - start,) + bank.shape[1:], dtype=np.float32)
            for rank in range(idx_train.shape[1]):
                source_ids = valid_ids[idx_train[start:end, rank]]
                train_block += w_train[start:end, rank].reshape((end - start,) + train_shape) * np.asarray(
                    bank[source_ids], dtype=np.float32
                )
            test_block = np.sum(
                current[idx_test[start:end]]
                * w_test[start:end].reshape((end - start, idx_test.shape[1]) + train_shape),
                axis=1,
            ).astype(np.float32)
            nxt[start:end] = normalize_full(train_mix * train_block + (1.0 - train_mix) * test_block)
        current = nxt
        print(f"v224 harmonic {name} mix={train_mix:.2f} iter={iteration}/{iterations}", flush=True)
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=current.shape)
    out[:] = current
    out.flush()
    return np.load(output_path, mmap_mode="r")


def run_variant(
    name: str,
    leaked_target: np.ndarray,
    delay_target: np.ndarray,
    regimes: np.ndarray,
    leaked_alpha: dict[str, float],
    delay_alpha: dict[str, float],
) -> dict[str, object]:
    leaked_path = OUT_DIR / f"v224_{name}_after_leaked.npy"
    output_path = OUT_DIR / f"v224_{name}_Round1_Test_Channel.npy"
    leaked_diag = v223.apply_leaked_full(v223.BASE, leaked_path, leaked_target, leaked_alpha, regimes)
    delay_diag = v223.apply_delay_full(leaked_path, output_path, delay_target, delay_alpha, regimes)
    score = v208.score_prediction(output_path)
    return {
        "name": name,
        "output": PATHS.relative(output_path),
        "leaked_alpha": leaked_alpha,
        "delay_alpha": delay_alpha,
        "score": score,
        "diagnostics": {
            "leaked": {key: summarize(value) for key, value in leaked_diag.items()},
            "delay": {key: summarize(value) for key, value in delay_diag.items()},
        },
    }


def main() -> None:
    leaked_support, delay_support, valid = v223.build_train_supports()
    leaked_init, delay_init, base_target_diag = v223.build_test_targets(leaked_support, delay_support, valid)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    valid_ids = np.flatnonzero(valid).astype(np.int64)
    idx_train, w_train, idx_test, w_test = train_test_weights(
        train_pos,
        train_blocked,
        test_pos,
        test_blocked,
        valid_ids,
    )
    regimes = regime_of(test_blocked)

    propagated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for train_mix, iterations in ((0.72, 5), (0.58, 8)):
        leaked = harmonic_support(
            "leaked",
            leaked_support,
            valid_ids,
            idx_train,
            w_train,
            idx_test,
            w_test,
            leaked_init,
            train_mix,
            iterations,
        )
        delay = harmonic_support(
            "delay",
            delay_support,
            valid_ids,
            idx_train,
            w_train,
            idx_test,
            w_test,
            delay_init,
            train_mix,
            iterations,
        )
        propagated[f"mix{train_mix:.2f}_it{iterations}"] = (leaked, delay)

    variants = []
    for key, (leaked_target, delay_target) in propagated.items():
        variants.append(
            run_variant(
                f"{key}_moderate",
                leaked_target,
                delay_target,
                regimes,
                {"clear": 0.0, "light": 0.04, "medium": 0.26, "heavy": 0.36},
                {"clear": 0.0, "light": 0.06, "medium": 0.34, "heavy": 0.48},
            )
        )
        variants.append(
            run_variant(
                f"{key}_delay_only",
                leaked_init,
                delay_target,
                regimes,
                {"clear": 0.0, "light": 0.0, "medium": 0.0, "heavy": 0.0},
                {"clear": 0.0, "light": 0.08, "medium": 0.58, "heavy": 0.82},
            )
        )

    best = max(variants, key=lambda item: item["score"]["score_traceback_leaked"])
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": "Official train-H full tensor labels become more useful when propagated over the combined train/test graph instead of direct train->test KNN.",
        "compliance_boundary": {
            "generation_uses": [
                "official train H",
                "official train/test positions",
                "official geometry caches",
                "v209 official-data backbone",
            ],
            "generation_does_not_use": [
                "Sionna generated train/test labels",
                "v212/v217 teacher outputs",
                "hidden labels",
                "synthetic memory channels",
            ],
            "note": "Analysis-only deterministic graph propagation; not a recommended AI candidate.",
        },
        "base_target_diagnostics": base_target_diag,
        "graph": {
            "k_train": K_TRAIN,
            "k_test": K_TEST,
            "train_weight_effective_count": summarize(1.0 / np.maximum(np.sum(w_train**2, axis=1), TINY)),
            "test_weight_effective_count": summarize(1.0 / np.maximum(np.sum(w_test**2, axis=1), TINY)),
        },
        "variants": variants,
        "best_variant": best["name"],
        "best_score_traceback_leaked": best["score"]["score_traceback_leaked"],
        "local_gate_required": 0.645604,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
