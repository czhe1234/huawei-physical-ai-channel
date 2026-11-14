#!/usr/bin/env python3
"""v223 analysis: official train-H full-tensor support signal.

This is an analysis-only probe. It uses only official train H, official
train/test coordinates, and official point-cloud-derived geometry caches to
construct full leaked-PAS and delay-domain support targets. It does not use
Sionna generated train/test labels, v212/v217 teacher outputs, or hidden test
truth for generation. The local v208 scorer is used only after generation as a
diagnostic gate.

The point of the probe is to check whether the full tensor structure in
official train H contains more useful signal than the compressed marginals used
by v218. If this is positive, the next step is to replace this diagnostic KNN
transfer with a neural masked-spectrum model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import analyze_v208_strict_historical_replay as v208
import generate_v187_leaked_pas_lifecycle_projection as leak
import generate_v196_train_guided_delay_pdp_mouth as v196
import generate_v218_official_multispectrum_physics_ai as v218
from generate_v87_easy_field_projection import MP, MH, MV, N_ANT, S_SC, summarize
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 223
VERSION = "v223_official_fulltensor_support_signal"

OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / "v223_official_fulltensor_support_signal.json"
TRAIN_LEAKED = OUT_DIR / "v223_train_full_leaked_support.npy"
TRAIN_DELAY = OUT_DIR / "v223_train_full_delay_support.npy"
TEST_LEAKED_TARGET = OUT_DIR / "v223_test_full_leaked_target.npy"
TEST_DELAY_TARGET = OUT_DIR / "v223_test_full_delay_target.npy"

BASE = PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy"
GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

BATCH = 4
AVG_BATCH = 3
PROJ_BATCH = 3
FLOOR = 1e-12


def normalize_full(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.reshape(len(out), -1).sum(axis=1), TINY).reshape(
        (len(out),) + (1,) * (out.ndim - 1)
    )
    return out.astype(np.float32)


def denoise_power_support(power: np.ndarray, strength: float) -> np.ndarray:
    flat = np.maximum(power.reshape(len(power), -1).astype(np.float32), 0.0)
    total = flat.sum(axis=1)
    valid = total > 1e-20
    floor = np.quantile(flat, 0.20, axis=1).astype(np.float32)
    clean = np.maximum(flat - strength * floor[:, None], 0.0)
    empty = clean.sum(axis=1) <= 1e-20
    clean[empty] = flat[empty]
    clean[~valid] = 0.0
    support = clean.reshape(power.shape)
    denom = np.maximum(support.reshape(len(support), -1).sum(axis=1), TINY)
    support = support / denom.reshape((len(support),) + (1,) * (support.ndim - 1))
    support[~valid] = 0.0
    return support.astype(np.float32)


def build_train_supports() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_h = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    n = len(train_h)
    if TRAIN_LEAKED.is_file() and TRAIN_DELAY.is_file():
        leaked_support = np.load(TRAIN_LEAKED, mmap_mode="r")
        delay_support = np.load(TRAIN_DELAY, mmap_mode="r")
        valid = np.sum(np.asarray(leaked_support).reshape(n, -1), axis=1) > 0.5
        return leaked_support, delay_support, valid

    leaked_support = np.lib.format.open_memmap(
        TRAIN_LEAKED,
        mode="w+",
        dtype=np.float32,
        shape=(n, MP, S_SC, MH, N_ANT, MV),
    )
    delay_support = np.lib.format.open_memmap(
        TRAIN_DELAY,
        mode="w+",
        dtype=np.float32,
        shape=(n, MP, MH, MV, N_ANT, S_SC),
    )
    valid = np.zeros(n, dtype=bool)
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        h = np.asarray(train_h[start:end], dtype=np.complex64)
        leaked_power = leak.leaked_cell_power(leak.leaked_forward(h))
        delay_power = v196.delay_power_from_h(h)
        total = np.sum(np.abs(h) ** 2, axis=(1, 2, 3), dtype=np.float64)
        valid[start:end] = total > 1e-20
        leaked_support[start:end] = denoise_power_support(leaked_power, 0.60)
        delay_support[start:end] = denoise_power_support(delay_power, 0.55)
        leaked_support.flush()
        delay_support.flush()
        print(f"v223 train full supports {end}/{n}", flush=True)
    return np.load(TRAIN_LEAKED, mmap_mode="r"), np.load(TRAIN_DELAY, mmap_mode="r"), valid


def weighted_full_average(
    bank: np.ndarray,
    global_idx: np.ndarray,
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
    for start in range(0, len(neighbor_idx), AVG_BATCH):
        end = min(start + AVG_BATCH, len(neighbor_idx))
        block = np.zeros((end - start,) + bank.shape[1:], dtype=np.float32)
        for rank in range(neighbor_idx.shape[1]):
            source_ids = global_idx[neighbor_idx[start:end, rank]]
            block += weights[start:end, rank].reshape(
                (end - start,) + (1,) * (bank.ndim - 1)
            ) * np.asarray(bank[source_ids], dtype=np.float32)
        out[start:end] = normalize_full(block)
        out.flush()
        print(f"v223 weighted full average {output_path.name} {end}/{len(neighbor_idx)}", flush=True)
    return np.load(output_path, mmap_mode="r")


def build_test_targets(
    leaked_support: np.ndarray,
    delay_support: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    valid_ids = np.flatnonzero(valid).astype(np.int64)
    idx, weights = v218.neighbor_index_weights(
        test_pos,
        test_blocked,
        train_pos[valid_ids],
        train_blocked[valid_ids],
        k=72,
    )
    leaked_target = weighted_full_average(leaked_support, valid_ids, idx, weights, TEST_LEAKED_TARGET)
    delay_target = weighted_full_average(delay_support, valid_ids, idx, weights, TEST_DELAY_TARGET)
    diagnostics = {
        "valid_train_count": int(len(valid_ids)),
        "neighbor_k": int(idx.shape[1]),
        "weight_effective_count": summarize(1.0 / np.maximum(np.sum(weights**2, axis=1), TINY)),
        "nearest_train_distance_proxy": summarize(1.0 / np.maximum(weights[:, 0], TINY)),
    }
    return leaked_target, delay_target, diagnostics


def apply_leaked_full(
    source_path: Path,
    output_path: Path,
    target: np.ndarray,
    alpha_by_regime: dict[str, float],
    regimes: np.ndarray,
) -> dict[str, np.ndarray]:
    source = np.load(source_path, mmap_mode="r")
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.complex64, shape=source.shape)
    diag: dict[str, list[np.ndarray]] = {
        "leaked_l1": [],
        "leaked_gain_min": [],
        "leaked_gain_p90": [],
        "leaked_power_factor": [],
    }
    alpha = np.asarray([alpha_by_regime[str(r)] for r in regimes], dtype=np.float32)
    for start in range(0, len(source), PROJ_BATCH):
        end = min(start + PROJ_BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        spec0 = leak.leaked_forward(h0)
        power0 = leak.leaked_cell_power(spec0)
        total0 = np.maximum(power0.reshape(end - start, -1).sum(axis=1), TINY).astype(np.float32)
        support0 = power0 / total0.reshape((end - start, 1, 1, 1, 1, 1))
        shape = (end - start,) + (1,) * (support0.ndim - 1)
        desired = normalize_full(
            (1.0 - alpha[start:end].reshape(shape)) * support0
            + alpha[start:end].reshape(shape) * np.asarray(target[start:end], dtype=np.float32)
        )
        gain = np.sqrt(desired / np.maximum(support0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.20, 5.80)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.50, 2.35)
        gain[strong1024] = np.clip(gain[strong1024], 0.66, 1.68)
        gain[strong128] = np.clip(gain[strong128], 0.82, 1.28)
        h1 = leak.leaked_inverse(spec0 * gain)
        p0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
        p1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
        h1 *= np.sqrt(p0 / p1)[:, None, None, None].astype(np.float32)
        out[start:end] = h1.astype(np.complex64)
        out.flush()
        p_after = leak.leaked_cell_power(leak.leaked_forward(h1.astype(np.complex64)))
        s_after = p_after / np.maximum(
            p_after.reshape(end - start, -1).sum(axis=1), TINY
        ).reshape((end - start, 1, 1, 1, 1, 1))
        diag["leaked_l1"].append(np.sum(np.abs(s_after - support0), axis=(1, 2, 3, 4, 5)).astype(np.float32))
        diag["leaked_gain_min"].append(gain.reshape(end - start, -1).min(axis=1).astype(np.float32))
        diag["leaked_gain_p90"].append(np.quantile(gain.reshape(end - start, -1), 0.90, axis=1).astype(np.float32))
        diag["leaked_power_factor"].append(
            (np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / p0).astype(np.float32)
        )
        print(f"v223 leaked full projection {output_path.name} {end}/{len(source)}", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def apply_delay_full(
    source_path: Path,
    output_path: Path,
    target: np.ndarray,
    alpha_by_regime: dict[str, float],
    regimes: np.ndarray,
) -> dict[str, np.ndarray]:
    source = np.load(source_path, mmap_mode="r")
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.complex64, shape=source.shape)
    diag: dict[str, list[np.ndarray]] = {
        "delay_l1": [],
        "delay_gain_min": [],
        "delay_gain_p90": [],
        "delay_power_factor": [],
    }
    alpha = np.asarray([alpha_by_regime[str(r)] for r in regimes], dtype=np.float32)
    for start in range(0, len(source), PROJ_BATCH):
        end = min(start + PROJ_BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        cir0 = np.fft.ifft(h0, axis=-1)
        arr0 = cir0.reshape(end - start, MP, MH, MV, N_ANT, S_SC)
        power0 = (np.abs(arr0).astype(np.float32) ** 2).astype(np.float32)
        total0 = np.maximum(power0.reshape(end - start, -1).sum(axis=1), TINY).astype(np.float32)
        support0 = power0 / total0.reshape((end - start, 1, 1, 1, 1, 1))
        shape = (end - start,) + (1,) * (support0.ndim - 1)
        desired = normalize_full(
            (1.0 - alpha[start:end].reshape(shape)) * support0
            + alpha[start:end].reshape(shape) * np.asarray(target[start:end], dtype=np.float32)
        )
        gain = np.sqrt(desired / np.maximum(support0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.24, 4.40)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.54, 2.10)
        gain[strong1024] = np.clip(gain[strong1024], 0.70, 1.52)
        gain[strong128] = np.clip(gain[strong128], 0.86, 1.22)
        h1 = np.fft.fft((arr0 * gain).reshape(end - start, 256, N_ANT, S_SC), axis=-1).astype(np.complex64)
        p0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
        p1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
        h1 *= np.sqrt(p0 / p1)[:, None, None, None].astype(np.float32)
        out[start:end] = h1.astype(np.complex64)
        out.flush()
        p_after = v196.delay_power_from_h(h1.astype(np.complex64))
        s_after = p_after / np.maximum(
            p_after.reshape(end - start, -1).sum(axis=1), TINY
        ).reshape((end - start, 1, 1, 1, 1, 1))
        diag["delay_l1"].append(np.sum(np.abs(s_after - support0), axis=(1, 2, 3, 4, 5)).astype(np.float32))
        diag["delay_gain_min"].append(gain.reshape(end - start, -1).min(axis=1).astype(np.float32))
        diag["delay_gain_p90"].append(np.quantile(gain.reshape(end - start, -1), 0.90, axis=1).astype(np.float32))
        diag["delay_power_factor"].append(
            (np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / p0).astype(np.float32)
        )
        print(f"v223 delay full projection {output_path.name} {end}/{len(source)}", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def run_variant(
    name: str,
    leaked_target: np.ndarray,
    delay_target: np.ndarray,
    regimes: np.ndarray,
    leaked_alpha: dict[str, float],
    delay_alpha: dict[str, float],
) -> dict[str, object]:
    leaked_path = OUT_DIR / f"v223_{name}_after_leaked.npy"
    output_path = OUT_DIR / f"v223_{name}_Round1_Test_Channel.npy"
    leaked_diag = apply_leaked_full(BASE, leaked_path, leaked_target, leaked_alpha, regimes)
    delay_diag = apply_delay_full(leaked_path, output_path, delay_target, delay_alpha, regimes)
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
    for path in [BASE, GEOM_CACHE, PATHS.round1 / "Round1_Train_Channel.npy"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    leaked_support, delay_support, valid = build_train_supports()
    leaked_target, delay_target, target_diag = build_test_targets(leaked_support, delay_support, valid)
    geom = np.load(GEOM_CACHE)
    regimes = regime_of(geom["test_blocked"].astype(np.float32))
    variants = [
        (
            "moderate",
            {"clear": 0.0, "light": 0.04, "medium": 0.26, "heavy": 0.36},
            {"clear": 0.0, "light": 0.06, "medium": 0.34, "heavy": 0.48},
        ),
        (
            "strong",
            {"clear": 0.0, "light": 0.06, "medium": 0.42, "heavy": 0.58},
            {"clear": 0.0, "light": 0.08, "medium": 0.56, "heavy": 0.76},
        ),
        (
            "delay_only_strong",
            {"clear": 0.0, "light": 0.0, "medium": 0.0, "heavy": 0.0},
            {"clear": 0.0, "light": 0.08, "medium": 0.58, "heavy": 0.82},
        ),
    ]
    results = [
        run_variant(name, leaked_target, delay_target, regimes, leak_a, delay_a)
        for name, leak_a, delay_a in variants
    ]
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": "Official train H full leaked/delay tensor support contains middle-weak propagation texture lost by v218 marginal compression.",
        "compliance_boundary": {
            "generation_uses": [
                "official Round1_Train_Channel.npy",
                "official train/test positions",
                "official point-cloud-derived geometry caches",
                "pre-Sionna v209 official-data backbone",
            ],
            "generation_does_not_use": [
                "Sionna generated train/test labels",
                "v212/v217 teacher outputs",
                "hidden online labels",
                "synthetic memory channels",
            ],
            "note": "This analysis probe itself is KNN full-tensor transfer, so it is not a recommended AI candidate. It is used only to decide whether to build a neural masked-spectrum model next.",
        },
        "target_diagnostics": target_diag,
        "variants": results,
        "best_variant": max(results, key=lambda item: item["score"]["score_traceback_leaked"])["name"],
        "local_gate_required": 0.645604,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
