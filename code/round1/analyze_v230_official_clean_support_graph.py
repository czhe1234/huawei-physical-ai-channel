#!/usr/bin/env python3
"""v230 analysis: stronger official train-H clean support graph.

v223/v224 used full-tensor official train-H support with a moderate
floor-subtraction target.  v230 tests whether the released train labels are
still too noisy for support propagation: build a more aggressively cleaned
full leaked/delay support target from official train H, then run the same
strict official train/test graph harmonic propagation.

No Sionna generated labels, v212/v217 teachers, hidden labels, or synthetic
truth are used for generation.  The v208 scorer is only the post-generation
hard gate diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v223_official_fulltensor_support_signal as v223
import analyze_v224_official_graph_harmonic_support as v224
import generate_v187_leaked_pas_lifecycle_projection as leak
import generate_v196_train_guided_delay_pdp_mouth as v196
from generate_v87_easy_field_projection import MP, MH, MV, N_ANT, S_SC, summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 230
VERSION = "v230_official_clean_support_graph"

OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v230_official_clean_support_graph.json"

GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"
BASE = PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy"

BATCH = 20
AVG_BATCH = 3
FLOOR = 1e-12
LOCAL_GATE = 0.645604


def normalize_full(x: np.ndarray) -> np.ndarray:
    return v223.normalize_full(x)


def denoise_power_support(power: np.ndarray, strength: float, q: float) -> np.ndarray:
    flat = np.maximum(power.reshape(len(power), -1).astype(np.float32), 0.0)
    total = flat.sum(axis=1)
    valid = total > 1e-20
    floor = np.quantile(flat, q, axis=1).astype(np.float32)
    clean = np.maximum(flat - strength * floor[:, None], 0.0)
    # Protect very strong cells from an over-aggressive global subtraction by
    # mixing the top-128 support back before normalization.
    k = min(128, flat.shape[1])
    ids = np.argpartition(flat, -k, axis=1)[:, -k:]
    rows = np.arange(len(flat))[:, None]
    clean[rows, ids] = np.maximum(clean[rows, ids], flat[rows, ids] - 0.25 * floor[:, None])
    empty = clean.sum(axis=1) <= 1e-20
    clean[empty] = flat[empty]
    clean[~valid] = 0.0
    support = clean.reshape(power.shape)
    denom = np.maximum(support.reshape(len(support), -1).sum(axis=1), TINY)
    support = support / denom.reshape((len(support),) + (1,) * (support.ndim - 1))
    support[~valid] = 0.0
    return support.astype(np.float32)


def build_clean_supports(leak_strength: float, delay_strength: float, q: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tag = f"ls{leak_strength:.2f}_ds{delay_strength:.2f}_q{q:.2f}".replace(".", "p")
    leaked_path = OUT_DIR / f"v230_train_full_leaked_{tag}.npy"
    delay_path = OUT_DIR / f"v230_train_full_delay_{tag}.npy"
    done_path = OUT_DIR / f"v230_train_full_support_{tag}.done"
    train_h = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    n = len(train_h)
    if leaked_path.is_file() and delay_path.is_file() and done_path.is_file():
        leaked_support = np.load(leaked_path, mmap_mode="r")
        delay_support = np.load(delay_path, mmap_mode="r")
        valid = np.sum(np.asarray(leaked_support).reshape(n, -1), axis=1) > 0.5
        return leaked_support, delay_support, valid

    leaked_support = np.lib.format.open_memmap(
        leaked_path,
        mode="w+",
        dtype=np.float32,
        shape=(n, MP, S_SC, MH, N_ANT, MV),
    )
    delay_support = np.lib.format.open_memmap(
        delay_path,
        mode="w+",
        dtype=np.float32,
        shape=(n, MP, MH, MV, N_ANT, S_SC),
    )
    valid = np.zeros(n, dtype=bool)
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        h = np.asarray(train_h[start:end], dtype=np.complex64)
        total = np.sum(np.abs(h) ** 2, axis=(1, 2, 3), dtype=np.float64)
        valid[start:end] = total > 1e-20
        leaked_support[start:end] = denoise_power_support(
            leak.leaked_cell_power(leak.leaked_forward(h)),
            leak_strength,
            q,
        )
        delay_support[start:end] = denoise_power_support(
            v196.delay_power_from_h(h),
            delay_strength,
            q,
        )
        leaked_support.flush()
        delay_support.flush()
        if end == n or end % 100 == 0:
            print(f"v230 clean train support {end}/{n}", flush=True)
    done_path.write_text("complete\n", encoding="utf-8")
    return np.load(leaked_path, mmap_mode="r"), np.load(delay_path, mmap_mode="r"), valid


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
    output_path = OUT_DIR / f"v230_{name}_mix{train_mix:.2f}_it{iterations}.npy"
    if output_path.is_file():
        return np.load(output_path, mmap_mode="r")
    current = np.asarray(init, dtype=np.float32).copy()
    shape = (1,) * (bank.ndim - 1)
    for iteration in range(1, iterations + 1):
        nxt = np.empty_like(current)
        for start in range(0, len(current), AVG_BATCH):
            end = min(start + AVG_BATCH, len(current))
            train_block = np.zeros((end - start,) + bank.shape[1:], dtype=np.float32)
            for rank in range(idx_train.shape[1]):
                source_ids = valid_ids[idx_train[start:end, rank]]
                train_block += w_train[start:end, rank].reshape((end - start,) + shape) * np.asarray(
                    bank[source_ids],
                    dtype=np.float32,
                )
            test_block = np.sum(
                current[idx_test[start:end]]
                * w_test[start:end].reshape((end - start, idx_test.shape[1]) + shape),
                axis=1,
            ).astype(np.float32)
            nxt[start:end] = normalize_full(train_mix * train_block + (1.0 - train_mix) * test_block)
        current = nxt
        print(f"v230 harmonic {name} mix={train_mix:.2f} iter={iteration}/{iterations}", flush=True)
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=current.shape)
    out[:] = current
    out.flush()
    return np.load(output_path, mmap_mode="r")


def weighted_full_average(
    bank: np.ndarray,
    valid_ids: np.ndarray,
    neighbor_idx: np.ndarray,
    weights: np.ndarray,
    output_path: Path,
) -> np.ndarray:
    if output_path.is_file():
        return np.load(output_path, mmap_mode="r")
    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(neighbor_idx),) + bank.shape[1:],
    )
    shape = (1,) * (bank.ndim - 1)
    for start in range(0, len(neighbor_idx), AVG_BATCH):
        end = min(start + AVG_BATCH, len(neighbor_idx))
        block = np.zeros((end - start,) + bank.shape[1:], dtype=np.float32)
        for rank in range(neighbor_idx.shape[1]):
            source_ids = valid_ids[neighbor_idx[start:end, rank]]
            block += weights[start:end, rank].reshape((end - start,) + shape) * np.asarray(
                bank[source_ids],
                dtype=np.float32,
            )
        out[start:end] = normalize_full(block)
        out.flush()
        if end == len(neighbor_idx) or end % 60 == 0:
            print(f"v230 weighted average {output_path.name} {end}/{len(neighbor_idx)}", flush=True)
    return np.load(output_path, mmap_mode="r")


def build_test_targets(
    leaked_support: np.ndarray,
    delay_support: np.ndarray,
    valid: np.ndarray,
    tag: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    valid_ids = np.flatnonzero(valid).astype(np.int64)
    idx, weights = v224.v218.neighbor_index_weights(
        test_pos,
        test_blocked,
        train_pos[valid_ids],
        train_blocked[valid_ids],
        k=72,
    )
    leaked_target = weighted_full_average(
        leaked_support,
        valid_ids,
        idx,
        weights,
        OUT_DIR / f"v230_test_full_leaked_target_{tag}.npy",
    )
    delay_target = weighted_full_average(
        delay_support,
        valid_ids,
        idx,
        weights,
        OUT_DIR / f"v230_test_full_delay_target_{tag}.npy",
    )
    diagnostics = {
        "valid_train_count": int(len(valid_ids)),
        "neighbor_k": int(idx.shape[1]),
        "weight_effective_count": summarize(1.0 / np.maximum(np.sum(weights**2, axis=1), TINY)),
        "nearest_train_distance_proxy": summarize(1.0 / np.maximum(weights[:, 0], TINY)),
    }
    return leaked_target, delay_target, diagnostics


def run_variant(
    name: str,
    leaked_target: np.ndarray,
    delay_target: np.ndarray,
    regimes: np.ndarray,
    leaked_alpha: dict[str, float],
    delay_alpha: dict[str, float],
) -> dict[str, object]:
    leaked_path = OUT_DIR / f"v230_{name}_after_leaked.npy"
    output_path = OUT_DIR / f"v230_{name}_Round1_Test_Channel.npy"
    leaked_diag = v223.apply_leaked_full(BASE, leaked_path, leaked_target, leaked_alpha, regimes)
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
    for path in [BASE, GEOM_CACHE]:
        if not path.is_file():
            raise FileNotFoundError(path)

    leaked_support, delay_support, valid = build_clean_supports(0.95, 0.90, 0.24)
    target_tag = "ls0p95_ds0p90_q0p24"
    leaked_init, delay_init, base_diag = build_test_targets(leaked_support, delay_support, valid, target_tag)

    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    valid_ids = np.flatnonzero(valid).astype(np.int64)
    idx_train, w_train, idx_test, w_test = v224.train_test_weights(
        train_pos,
        train_blocked,
        test_pos,
        test_blocked,
        valid_ids,
    )
    regimes = regime_of(test_blocked)

    leaked_target = harmonic_support(
        "leaked_clean",
        leaked_support,
        valid_ids,
        idx_train,
        w_train,
        idx_test,
        w_test,
        leaked_init,
        0.58,
        8,
    )
    delay_target = harmonic_support(
        "delay_clean",
        delay_support,
        valid_ids,
        idx_train,
        w_train,
        idx_test,
        w_test,
        delay_init,
        0.58,
        8,
    )

    variants = [
        run_variant(
            "clean095_moderate",
            leaked_target,
            delay_target,
            regimes,
            {"clear": 0.0, "light": 0.035, "medium": 0.24, "heavy": 0.34},
            {"clear": 0.0, "light": 0.055, "medium": 0.32, "heavy": 0.46},
        ),
        run_variant(
            "clean095_delay_strong",
            leaked_init,
            delay_target,
            regimes,
            {"clear": 0.0, "light": 0.0, "medium": 0.0, "heavy": 0.0},
            {"clear": 0.0, "light": 0.075, "medium": 0.54, "heavy": 0.78},
        ),
    ]
    best = max(variants, key=lambda item: item["score"]["score_traceback_leaked"])
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Official train H contains additive-noise support floor; stronger clean full-support "
            "targets may improve graph harmonic propagation."
        ),
        "compliance_boundary": {
            "generation_uses": [
                "official train H",
                "official train/test positions",
                "official geometry cache",
                "v209 official-data backbone",
            ],
            "generation_does_not_use": [
                "Sionna generated channel labels",
                "v212/v217 teacher outputs",
                "hidden online labels",
                "synthetic replay truth",
            ],
        },
        "local_gate_required": LOCAL_GATE,
        "cleaning": {"leaked_strength": 0.95, "delay_strength": 0.90, "floor_quantile": 0.24},
        "base_target_diagnostics": base_diag,
        "graph": {
            "train_mix": 0.58,
            "iterations": 8,
            "k_train": v224.K_TRAIN,
            "k_test": v224.K_TEST,
        },
        "variants": variants,
        "best_variant": best["name"],
        "best_score_traceback_leaked": best["score"]["score_traceback_leaked"],
        "gate_passed": bool(best["score"]["score_traceback_leaked"] >= LOCAL_GATE),
        "outputs": {"report": PATHS.relative(REPORT)},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
