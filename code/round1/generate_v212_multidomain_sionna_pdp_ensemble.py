#!/usr/bin/env python3
"""Generate v212: multi-domain Sionna PDP ensemble on top of v211.

v210 and v211 produced the largest recent online jumps by using the
high-budget v207 Sionna normalized PDP/delay shape.  v211 was still a
single-reference continuation.  v212 adds a new information dimension: the
lower-budget v205 Sionna full-H domains at 2.6, 6.0 and 6.7 GHz.

The intent is not to copy those weaker domains.  They are used as a stability
check and a soft ensemble around the v207 high-budget target:

* if multi-frequency delay support agrees with v207, continue moving v211;
* if the low-budget domains disagree or are zero, damp the movement;
* preserve v211 point power and delay-domain phase exactly.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

import generate_v178_test_track_complex_consensus as track
import generate_v196_train_guided_delay_pdp_mouth as v196
from generate_v87_easy_field_projection import MH, N_ANT, S_SC, summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v212_multidomain_sionna_pdp_ensemble"

OUT_DIR = PATHS.cache_dir(212)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(212)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v212_Round1_Test_Channel.npy"
MARGINALS = OUT_DIR / "v212_multidomain_sionna_pdp_marginals.npz"
ACTION_OUT = OUT_DIR / "v212_multidomain_sionna_pdp_actions.npz"
REPORT = REPORT_DIR / "v212_multidomain_sionna_pdp_ensemble_generate.json"
ZIP_OUT = PATHS.round1_versions / "v212_multidomain_sionna_pdp_ensemble_candidate_pending.zip"

BASE = PATHS.cache_dir(211) / "v211_Round1_Test_Channel.npy"
V207_REFERENCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
V196_TARGET = PATHS.cache_dir(196) / "v196_train_guided_delay_target.npy"
V211_ACTIONS = PATHS.cache_dir(211) / "v211_sionna_pdp_completion_actions.npz"
V210_ACTIONS = PATHS.cache_dir(210) / "v210_sionna_pdp_prior_actions.npz"
V211_BASE = PATHS.cache_dir(211) / "v211_Round1_Test_Channel.npy"
V210_BASE = PATHS.cache_dir(210) / "v210_Round1_Test_Channel.npy"
V209_BASE = PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy"

V205_REFERENCES = [
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f2p6" / "test_clean_H.npy",
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f6p0" / "test_clean_H.npy",
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f6p7" / "test_clean_H.npy",
]

MP = 2
MV = 8
FLOOR = 1e-12
BATCH = 6
IPF_ITERS = 7

D_ALPHA_BASE = {"clear": 0.0, "light": 0.010, "medium": 0.24, "heavy": 0.38}
D_ALPHA_CAP = {"clear": 0.0, "light": 0.028, "medium": 0.48, "heavy": 0.66}
DU_ALPHA_BASE = {"clear": 0.0, "light": 0.004, "medium": 0.062, "heavy": 0.108}
DU_ALPHA_CAP = {"clear": 0.0, "light": 0.012, "medium": 0.138, "heavy": 0.230}
DH_ALPHA_BASE = {"clear": 0.0, "light": 0.0035, "medium": 0.054, "heavy": 0.096}
DH_ALPHA_CAP = {"clear": 0.0, "light": 0.010, "medium": 0.122, "heavy": 0.208}

# v207 remains the primary reference.  v205 domains are lower-budget controls;
# they get weight only through agreement/stability, not raw replacement.
V207_WEIGHT_BASE = {"clear": 0.0, "light": 0.70, "medium": 0.82, "heavy": 0.88}
MULTI_WEIGHT_BASE = {"clear": 0.0, "light": 0.18, "medium": 0.30, "heavy": 0.34}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize(x: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=axis, keepdims=True), TINY)
    return out.astype(np.float32)


def safe_ratio(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    return (np.maximum(desired, FLOOR) / np.maximum(current, FLOOR)).astype(np.float32)


def delay_support_from_h(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    power = v196.delay_power_from_h(h)
    total = power.sum(axis=(1, 2, 3, 4, 5), dtype=np.float64).astype(np.float32)
    support = power / np.maximum(total[:, None, None, None, None, None], TINY)
    return support.astype(np.float32), total


def support_marginals(support: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = support.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
    du = support.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    dh = support.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)
    return normalize(d, axis=1), normalize(du, axis=(1, 2)), normalize(dh, axis=(1, 2))


def l1_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(a.reshape(len(a), -1) - b.reshape(len(b), -1)), axis=1).astype(np.float32)


def blend_valid_marginals(
    values: list[np.ndarray],
    valid: list[np.ndarray],
    axes: tuple[int, ...] | int,
) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(values, axis=0).astype(np.float32)
    valid_stack = np.stack(valid, axis=0).astype(np.float32)
    shape = (len(values), len(values[0])) + (1,) * (stack.ndim - 2)
    weights = valid_stack.reshape(shape)
    count = np.sum(valid_stack, axis=0).astype(np.float32)
    blended = np.sum(stack * weights, axis=0) / np.maximum(count.reshape((len(count),) + (1,) * (stack.ndim - 2)), 1.0)
    return normalize(blended, axis=axes), count


def build_marginals() -> dict[str, np.ndarray]:
    if MARGINALS.is_file():
        data = np.load(MARGINALS)
        return {key: data[key] for key in data.files}

    base = np.load(BASE, mmap_mode="r")
    v207 = np.load(V207_REFERENCE, mmap_mode="r")
    v196_target = np.load(V196_TARGET, mmap_mode="r")
    v205_refs = [np.load(path, mmap_mode="r") for path in V205_REFERENCES]
    n = len(base)
    arrays = {
        "current_d": np.empty((n, S_SC), dtype=np.float32),
        "current_du": np.empty((n, N_ANT, S_SC), dtype=np.float32),
        "current_dh": np.empty((n, MH, S_SC), dtype=np.float32),
        "v207_d": np.empty((n, S_SC), dtype=np.float32),
        "v207_du": np.empty((n, N_ANT, S_SC), dtype=np.float32),
        "v207_dh": np.empty((n, MH, S_SC), dtype=np.float32),
        "multi_d": np.empty((n, S_SC), dtype=np.float32),
        "multi_du": np.empty((n, N_ANT, S_SC), dtype=np.float32),
        "multi_dh": np.empty((n, MH, S_SC), dtype=np.float32),
        "v196_d": np.empty((n, S_SC), dtype=np.float32),
        "v196_du": np.empty((n, N_ANT, S_SC), dtype=np.float32),
        "v196_dh": np.empty((n, MH, S_SC), dtype=np.float32),
        "v207_total_power": np.empty(n, dtype=np.float32),
        "multi_valid_count": np.empty(n, dtype=np.float32),
        "multi_pair_l1": np.empty(n, dtype=np.float32),
    }

    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        current_support, _ = delay_support_from_h(np.asarray(base[start:end], dtype=np.complex64))
        v207_power = v196.delay_power_from_h(np.asarray(v207[start:end], dtype=np.complex64))
        v207_total = v207_power.sum(axis=(1, 2, 3, 4, 5), dtype=np.float64).astype(np.float32)
        v207_support = v207_power / np.maximum(v207_total[:, None, None, None, None, None], TINY)
        v207_zero = v207_total <= 1e-20
        if np.any(v207_zero):
            v207_support[v207_zero] = current_support[v207_zero]

        low_d: list[np.ndarray] = []
        low_du: list[np.ndarray] = []
        low_dh: list[np.ndarray] = []
        low_valid: list[np.ndarray] = []
        for ref in v205_refs:
            support, total = delay_support_from_h(np.asarray(ref[start:end], dtype=np.complex64))
            valid = total > 1e-20
            if np.any(~valid):
                support[~valid] = current_support[~valid]
            d, du, dh = support_marginals(support)
            low_d.append(d)
            low_du.append(du)
            low_dh.append(dh)
            low_valid.append(valid.astype(np.float32))

        multi_d, valid_count = blend_valid_marginals(low_d, low_valid, axes=1)
        multi_du, _ = blend_valid_marginals(low_du, low_valid, axes=(1, 2))
        multi_dh, _ = blend_valid_marginals(low_dh, low_valid, axes=(1, 2))
        pair_l1 = (
            l1_rows(low_d[0], low_d[1])
            + l1_rows(low_d[0], low_d[2])
            + l1_rows(low_d[1], low_d[2])
        ) / 3.0

        target_support = np.asarray(v196_target[start:end], dtype=np.float32)
        arrays["current_d"][start:end], arrays["current_du"][start:end], arrays["current_dh"][start:end] = support_marginals(current_support)
        arrays["v207_d"][start:end], arrays["v207_du"][start:end], arrays["v207_dh"][start:end] = support_marginals(v207_support)
        arrays["multi_d"][start:end] = multi_d
        arrays["multi_du"][start:end] = multi_du
        arrays["multi_dh"][start:end] = multi_dh
        arrays["v196_d"][start:end], arrays["v196_du"][start:end], arrays["v196_dh"][start:end] = support_marginals(target_support)
        arrays["v207_total_power"][start:end] = v207_total
        arrays["multi_valid_count"][start:end] = valid_count
        arrays["multi_pair_l1"][start:end] = pair_l1.astype(np.float32)
        print(f"v212 multidomain marginals {end}/{n}", flush=True)

    np.savez_compressed(MARGINALS, **arrays)
    data = np.load(MARGINALS)
    return {key: data[key] for key in data.files}


def build_actions(regimes: np.ndarray, marginals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    prior211 = np.load(V211_ACTIONS)
    prior210 = np.load(V210_ACTIONS)
    current_d = marginals["current_d"].astype(np.float32)
    current_du = marginals["current_du"].astype(np.float32)
    current_dh = marginals["current_dh"].astype(np.float32)
    v207_d = marginals["v207_d"].astype(np.float32)
    v207_du = marginals["v207_du"].astype(np.float32)
    v207_dh = marginals["v207_dh"].astype(np.float32)
    multi_d = marginals["multi_d"].astype(np.float32)
    multi_du = marginals["multi_du"].astype(np.float32)
    multi_dh = marginals["multi_dh"].astype(np.float32)
    v196_d = marginals["v196_d"].astype(np.float32)
    v196_du = marginals["v196_du"].astype(np.float32)
    v196_dh = marginals["v196_dh"].astype(np.float32)

    v207_zero = marginals["v207_total_power"].astype(np.float32) <= 1e-20
    valid_count = marginals["multi_valid_count"].astype(np.float32)
    pair_l1 = marginals["multi_pair_l1"].astype(np.float32)

    gap207 = l1_rows(v207_d, current_d)
    gap_multi = l1_rows(multi_d, current_d)
    gap207_multi = l1_rows(v207_d, multi_d)
    gap207_anchor = l1_rows(v207_d, v196_d)
    gap_multi_anchor = l1_rows(multi_d, v196_d)

    prev_alpha = prior211["alpha_d"].astype(np.float32)
    prev_full_l1 = prior211["full_l1_actual"].astype(np.float32)
    prev_d_l1 = prior211["d_l1_actual"].astype(np.float32)
    prev_mix = prior211["sionna_mix"].astype(np.float32)
    first_alpha = prior210["alpha_d"].astype(np.float32)

    multi_coverage = np.clip(valid_count / 3.0, 0.0, 1.0).astype(np.float32)
    multi_stability = np.clip(1.12 - 0.74 * pair_l1, 0.0, 1.0).astype(np.float32)
    cross_agreement = np.clip(1.06 - 0.55 * gap207_multi, 0.0, 1.0).astype(np.float32)
    anchor_agreement = np.clip(1.03 - 0.22 * gap207_anchor - 0.14 * gap_multi_anchor, 0.0, 1.0).astype(np.float32)
    remaining = np.clip((gap207 - 0.24) / 0.78, 0.0, 1.0).astype(np.float32)
    continuation = np.clip(0.42 + 0.36 * prev_alpha / 0.68 + 0.18 * first_alpha / 0.66, 0.0, 1.0)
    overpush_guard = np.clip((prev_full_l1 - 0.31) / 0.20 + (prev_d_l1 - 0.28) / 0.22, 0.0, 1.0)
    ensemble_quality = np.clip(
        0.22
        + 0.30 * cross_agreement
        + 0.22 * multi_stability
        + 0.18 * multi_coverage
        + 0.18 * anchor_agreement
        + 0.20 * remaining
        + 0.12 * continuation
        - 0.30 * overpush_guard,
        0.0,
        1.25,
    ).astype(np.float32)

    alpha_d = np.zeros(len(regimes), dtype=np.float32)
    alpha_du = np.zeros(len(regimes), dtype=np.float32)
    alpha_dh = np.zeros(len(regimes), dtype=np.float32)
    w207 = np.zeros(len(regimes), dtype=np.float32)
    wmulti = np.zeros(len(regimes), dtype=np.float32)
    for regime in ["clear", "light", "medium", "heavy"]:
        mask = regimes.astype(str) == regime
        if not np.any(mask):
            continue
        alpha_d[mask] = D_ALPHA_BASE[regime] * ensemble_quality[mask]
        alpha_du[mask] = DU_ALPHA_BASE[regime] * ensemble_quality[mask]
        alpha_dh[mask] = DH_ALPHA_BASE[regime] * ensemble_quality[mask]
        alpha_d[mask] = np.minimum(alpha_d[mask], D_ALPHA_CAP[regime])
        alpha_du[mask] = np.minimum(alpha_du[mask], DU_ALPHA_CAP[regime])
        alpha_dh[mask] = np.minimum(alpha_dh[mask], DH_ALPHA_CAP[regime])
        w207[mask] = V207_WEIGHT_BASE[regime] * np.clip(0.72 + 0.22 * cross_agreement[mask], 0.60, 1.0)
        wmulti[mask] = (
            MULTI_WEIGHT_BASE[regime]
            * multi_coverage[mask]
            * multi_stability[mask]
            * np.clip(0.72 + 0.30 * cross_agreement[mask], 0.55, 1.0)
        )

    strong_heavy = (
        (regimes.astype(str) == "heavy")
        & (remaining > 0.55)
        & (cross_agreement > 0.42)
        & (multi_stability > 0.42)
        & (multi_coverage >= 0.67)
        & (prev_alpha > 0.28)
    )
    alpha_d[strong_heavy] = np.minimum(alpha_d[strong_heavy] + 0.10, D_ALPHA_CAP["heavy"])
    alpha_du[strong_heavy] = np.minimum(alpha_du[strong_heavy] + 0.025, DU_ALPHA_CAP["heavy"])
    alpha_dh[strong_heavy] = np.minimum(alpha_dh[strong_heavy] + 0.022, DH_ALPHA_CAP["heavy"])
    w207[strong_heavy] = np.maximum(w207[strong_heavy], 0.84)
    wmulti[strong_heavy] = np.maximum(wmulti[strong_heavy], 0.22)

    block = (
        (regimes.astype(str) == "clear")
        | v207_zero
        | (cross_agreement < 0.18)
        | ((multi_coverage < 0.34) & (gap207_multi > 1.05))
        | (overpush_guard > 0.78)
    )
    alpha_d[block] = 0.0
    alpha_du[block] = 0.0
    alpha_dh[block] = 0.0
    w207[block] = 0.0
    wmulti[block] = 0.0

    w196 = np.maximum(0.0, 1.0 - w207 - wmulti).astype(np.float32)
    target_d = normalize(
        w196[:, None] * v196_d + w207[:, None] * v207_d + wmulti[:, None] * multi_d,
        axis=1,
    )
    target_du = normalize(
        w196[:, None, None] * v196_du + w207[:, None, None] * v207_du + wmulti[:, None, None] * multi_du,
        axis=(1, 2),
    )
    target_dh = normalize(
        w196[:, None, None] * v196_dh + w207[:, None, None] * v207_dh + wmulti[:, None, None] * multi_dh,
        axis=(1, 2),
    )
    desired_d = normalize((1.0 - alpha_d[:, None]) * current_d + alpha_d[:, None] * target_d, axis=1)
    desired_du = normalize((1.0 - alpha_du[:, None, None]) * current_du + alpha_du[:, None, None] * target_du, axis=(1, 2))
    desired_dh = normalize((1.0 - alpha_dh[:, None, None]) * current_dh + alpha_dh[:, None, None] * target_dh, axis=(1, 2))

    return {
        "regimes": regimes.astype("U8"),
        "alpha_d": alpha_d.astype(np.float32),
        "alpha_du": alpha_du.astype(np.float32),
        "alpha_dh": alpha_dh.astype(np.float32),
        "w207": w207.astype(np.float32),
        "wmulti": wmulti.astype(np.float32),
        "w196": w196.astype(np.float32),
        "multi_coverage": multi_coverage,
        "multi_stability": multi_stability,
        "cross_agreement": cross_agreement,
        "anchor_agreement": anchor_agreement,
        "remaining": remaining,
        "continuation": continuation.astype(np.float32),
        "overpush_guard": overpush_guard.astype(np.float32),
        "gap207": gap207,
        "gap_multi": gap_multi,
        "gap207_multi": gap207_multi,
        "gap207_anchor": gap207_anchor,
        "gap_multi_anchor": gap_multi_anchor,
        "prev_alpha_d": prev_alpha,
        "prev_mix": prev_mix,
        "desired_d": desired_d.astype(np.float32),
        "desired_du": desired_du.astype(np.float32),
        "desired_dh": desired_dh.astype(np.float32),
        "target_d": target_d.astype(np.float32),
        "target_du": target_du.astype(np.float32),
        "target_dh": target_dh.astype(np.float32),
    }


def apply_projection(actions: dict[str, np.ndarray], regimes: np.ndarray) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    desired_d = actions["desired_d"].astype(np.float32)
    desired_du = actions["desired_du"].astype(np.float32)
    desired_dh = actions["desired_dh"].astype(np.float32)
    diag: dict[str, list[np.ndarray]] = {
        "actual_power_factor": [],
        "full_l1_actual": [],
        "d_l1_actual": [],
        "du_l1_actual": [],
        "dh_l1_actual": [],
        "gain_min": [],
        "gain_p10": [],
        "gain_p90": [],
        "gain_max": [],
    }

    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        cir0 = np.fft.ifft(h0, axis=-1)
        arr0 = cir0.reshape(end - start, MP, MH, MV, N_ANT, S_SC)
        power0 = (np.abs(arr0).astype(np.float32) ** 2).astype(np.float32)
        total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        support0 = power0 / total0[:, None, None, None, None, None]
        h_power0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)

        work = support0.copy()
        for _ in range(IPF_ITERS):
            v196.apply_delay_ratio(
                work,
                safe_ratio(desired_d[start:end], work.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)),
                "d",
                0.99,
            )
            v196.apply_delay_ratio(
                work,
                safe_ratio(desired_du[start:end], work.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)),
                "du",
                0.34,
            )
            v196.apply_delay_ratio(
                work,
                safe_ratio(desired_dh[start:end], work.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)),
                "dh",
                0.31,
            )
            work = normalize(work.reshape(end - start, -1), axis=1).reshape(support0.shape)

        gain = np.sqrt((work * total0[:, None, None, None, None, None]) / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.18, 5.80).astype(np.float32)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.43, 2.45)
        gain[strong1024] = np.clip(gain[strong1024], 0.60, 1.70)
        gain[strong128] = np.clip(gain[strong128], 0.82, 1.30)

        for local, regime in enumerate(regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.18 * (gain[local] - 1.0)

        arr1 = arr0 * gain
        h1 = np.fft.fft(arr1.reshape(end - start, 256, N_ANT, S_SC), axis=-1)
        h_power1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        h1 *= np.sqrt(h_power0 / h_power1)[:, None, None, None].astype(np.float32)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()

        power1 = v196.delay_power_from_h(h1.astype(np.complex64))
        support1 = power1 / np.maximum(power1.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        diag["actual_power_factor"].append(
            (np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / h_power0).astype(np.float32)
        )
        diag["full_l1_actual"].append(np.sum(np.abs(support1 - support0), axis=(1, 2, 3, 4, 5)).astype(np.float32))
        diag["d_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 2, 3, 4), dtype=np.float64)
                    - support0.sum(axis=(1, 2, 3, 4), dtype=np.float64)
                ),
                axis=1,
            ).astype(np.float32)
        )
        diag["du_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 2, 3), dtype=np.float64)
                    - support0.sum(axis=(1, 2, 3), dtype=np.float64)
                ),
                axis=(1, 2),
            ).astype(np.float32)
        )
        diag["dh_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 3, 4), dtype=np.float64)
                    - support0.sum(axis=(1, 3, 4), dtype=np.float64)
                ),
                axis=(1, 2),
            ).astype(np.float32)
        )
        gflat = gain.reshape(end - start, -1)
        diag["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"v212 multidomain projection {end}/{len(source)}", flush=True)

    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for src in [
            PATHS.code / "generate_v212_multidomain_sionna_pdp_ensemble.py",
            PATHS.code / "generate_v211_sionna_pdp_completion_overdrive.py",
            PATHS.code / "generate_v210_sionna_pdp_prior_projection.py",
            PATHS.code / "generate_v207_sionna_test_reference.py",
            PATHS.code / "generate_v196_train_guided_delay_pdp_mouth.py",
            PATHS.code / "generate_v178_test_track_complex_consensus.py",
            PATHS.code / "generate_v87_easy_field_projection.py",
            PATHS.code / "generate_v171_full_complex_ad_tail_denoising.py",
            PATHS.code / "generate_v169_power_scaled_tail_denoising.py",
            PATHS.code / "generate_v159_frontier_consensus_support_soup.py",
            PATHS.code / "powerfield_common.py",
            PATHS.code_root / "project_paths.py",
        ]:
            archive.write(src, src.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")

    output = np.load(OUTPUT, mmap_mode="r")
    finite = True
    for start in range(0, len(output), 20):
        block = np.asarray(output[start : start + 20], dtype=np.complex64)
        finite = finite and bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    for path in [BASE, V207_REFERENCE, V196_TARGET, V211_ACTIONS, V210_ACTIONS, *V205_REFERENCES]:
        require_file(path)
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geom["test_blocked"].astype(np.float32))
    marginals = build_marginals()
    actions = build_actions(regimes, marginals)
    apply_diag = apply_projection(actions, regimes)
    action_diag = {**actions, **apply_diag}
    np.savez_compressed(ACTION_OUT, **action_diag)

    comparison_to_v211 = track.compare_output_to(OUTPUT, V211_BASE, regimes)
    comparison_to_v210 = track.compare_output_to(OUTPUT, V210_BASE, regimes)
    comparison_to_v209 = track.compare_output_to(OUTPUT, V209_BASE, regimes)
    if comparison_to_v211["shape"] != [500, 256, 4, 192] or comparison_to_v211["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v211['shape']} {comparison_to_v211['dtype']}")
    if not comparison_to_v211["finite"]:
        raise ValueError("v212 output contains non-finite values")

    support_active = actions["alpha_d"] > 1e-7
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v210/v211 prove high-budget Sionna normalized PDP/delay is the dominant target source. "
            "v212 adds multi-frequency v205 Sionna support as a stability ensemble so v211 can continue "
            "moving only where that signal is robust across carrier/domain changes."
        ),
        "new_information_used": {
            "v211_online_feedback": "v211=0.551384, +0.017235 vs v210, validates second-stage PDP completion",
            "multidomain_sionna": "v205 full-H references at 2.6, 6.0 and 6.7 GHz provide cross-frequency delay-support stability",
            "failure_boundary": "raw Sionna phase and raw Sionna total power are still not copied; point power is preserved",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "v207_reference": PATHS.relative(V207_REFERENCE),
            "v205_references": [PATHS.relative(path) for path in V205_REFERENCES],
            "v196_delay_target": PATHS.relative(V196_TARGET),
            "ipf_iters": IPF_ITERS,
            "d_alpha_base": D_ALPHA_BASE,
            "d_alpha_cap": D_ALPHA_CAP,
            "du_alpha_base": DU_ALPHA_BASE,
            "du_alpha_cap": DU_ALPHA_CAP,
            "dh_alpha_base": DH_ALPHA_BASE,
            "dh_alpha_cap": DH_ALPHA_CAP,
            "v207_weight_base": V207_WEIGHT_BASE,
            "multi_weight_base": MULTI_WEIGHT_BASE,
            "power_policy": "preserve v211 point power exactly",
            "phase_policy": "preserve v211 delay-domain phase; no raw Sionna phase transport",
        },
        "action_counts": {
            "support_active": int(np.sum(support_active)),
            "strong_heavy_count": int(
                np.sum(
                    (regimes.astype(str) == "heavy")
                    & (actions["remaining"] > 0.55)
                    & (actions["cross_agreement"] > 0.42)
                    & (actions["multi_stability"] > 0.42)
                    & (actions["multi_coverage"] >= 0.67)
                    & (actions["prev_alpha_d"] > 0.28)
                )
            ),
            "blocked_count": int(np.sum(actions["alpha_d"] <= 1e-7)),
        },
        "diagnostics": {
            "overall": {
                **{
                    key: summarize(value)
                    for key, value in action_diag.items()
                    if key
                    not in {
                        "regimes",
                        "desired_d",
                        "desired_du",
                        "desired_dh",
                        "target_d",
                        "target_du",
                        "target_dh",
                    }
                },
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "alpha_d": actions["alpha_d"],
                    "alpha_du": actions["alpha_du"],
                    "alpha_dh": actions["alpha_dh"],
                    "w207": actions["w207"],
                    "wmulti": actions["wmulti"],
                    "multi_coverage": actions["multi_coverage"],
                    "multi_stability": actions["multi_stability"],
                    "cross_agreement": actions["cross_agreement"],
                    "remaining": actions["remaining"],
                    "full_l1_actual": apply_diag["full_l1_actual"],
                    "d_l1_actual": apply_diag["d_l1_actual"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                },
            ),
        },
        "comparison_to_v211": comparison_to_v211,
        "comparison_to_v210": comparison_to_v210,
        "comparison_to_v209": comparison_to_v209,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "marginals": PATHS.relative(MARGINALS),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report = write_zip(report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
