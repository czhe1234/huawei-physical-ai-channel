#!/usr/bin/env python3
"""Generate v209: adaptive metric projection controller on top of v200.

This candidate revisits the current winning chain itself instead of importing
old residual-template machinery.  v189/v196/v200 showed that leaked PAS
marginals, delay/PDP marginals and selective total power are all real online
signals, but their strengths are still mostly hand-set.  v209 uses official
train truth only to build low-dimensional normalized spectral targets, then
lets a per-point controller decide how strongly v200 should move.

The design intentionally avoids the v155/v157 failure mode: it does not copy
full train-truth PAS/PDP slices or broad residuals.  Train truth is reduced to
stable marginals and is mixed with test-test consensus before projection.
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

import generate_v178_test_track_complex_consensus as track
import generate_v187_leaked_pas_lifecycle_projection as leak
import generate_v189_leaked_multimarginal_overdrive as v189
import generate_v196_train_guided_delay_pdp_mouth as v196
from generate_v87_easy_field_projection import MH, MP, MV, N_ANT, S_SC, summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v209_adaptive_metric_projection_controller"

OUT_DIR = PATHS.cache_dir(209)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(209)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v209_Round1_Test_Channel.npy"
INTERMEDIATE = OUT_DIR / "v209_after_leaked_projection.npy"
TRAIN_MARGINALS = OUT_DIR / "v209_train_leaked_low_marginals.npz"
CURRENT_SUPPORT = OUT_DIR / "v209_v200_leaked_full_support.npy"
LEAKED_TARGET = OUT_DIR / "v209_leaked_low_marginal_target.npy"
ACTION_OUT = OUT_DIR / "v209_adaptive_metric_projection_actions.npz"
REPORT = REPORT_DIR / "v209_adaptive_metric_projection_controller_generate.json"
ZIP_OUT = PATHS.round1_versions / "v209_adaptive_metric_projection_controller_candidate_pending.zip"

BASE = PATHS.cache_dir(200) / "v200_Round1_Test_Channel.npy"
V196_BASE = PATHS.cache_dir(196) / "v196_Round1_Test_Channel.npy"
V189_BASE = PATHS.cache_dir(189) / "v189_Round1_Test_Channel.npy"
V190_BASE = PATHS.cache_dir(190) / "v190_Round1_Test_Channel.npy"
V200_ACTIONS = PATHS.cache_dir(200) / "v200_selective_twosided_power_gate_actions.npz"
V199_DIAG = PATHS.cache_dir(199) / "v199_test_power_diagnostics.npz"
V196_ACTIONS = PATHS.cache_dir(196) / "v196_train_guided_delay_actions.npz"
V196_DELAY_TARGET = PATHS.cache_dir(196) / "v196_train_guided_delay_target.npy"

BATCH = 8
TRAIN_BATCH = 4
FLOOR = 1e-12
IPF_ITERS = 4

LEAK_ALPHA_BASE = {"clear": 0.0, "light": 0.055, "medium": 0.42, "heavy": 0.62}
LEAK_ALPHA_CAP = {"clear": 0.0, "light": 0.085, "medium": 0.56, "heavy": 0.76}
TRAIN_MIX_BASE = {"clear": 0.0, "light": 0.26, "medium": 0.44, "heavy": 0.50}

DELAY_EXTRA_BASE = {"clear": 0.0, "light": 0.025, "medium": 0.15, "heavy": 0.22}
DELAY_EXTRA_CAP = {"clear": 0.0, "light": 0.045, "medium": 0.22, "heavy": 0.30}

POWER_EXTRA_UP_CAP = {"clear": 0.025, "light": 0.014, "medium": 0.045, "heavy": 0.032}
POWER_EXTRA_DOWN_CAP = {"clear": -0.014, "light": -0.010, "medium": -0.040, "heavy": -0.055}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize(x: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=axis, keepdims=True), TINY)
    return out.astype(np.float32)


def safe_ratio(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    return (np.maximum(desired, FLOOR) / np.maximum(current, FLOOR)).astype(np.float32)


def build_train_leaked_marginals() -> dict[str, np.ndarray]:
    if TRAIN_MARGINALS.is_file():
        data = np.load(TRAIN_MARGINALS)
        return {key: data[key].astype(np.float32) for key in data.files}

    train_h = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    n = len(train_h)
    sh = np.empty((n, S_SC, MH), dtype=np.float32)
    psh = np.empty((n, MP, S_SC, MH), dtype=np.float32)
    shue = np.empty((n, S_SC, MH, N_ANT), dtype=np.float32)
    shv = np.empty((n, S_SC, MH, MV), dtype=np.float32)
    total_power = np.empty(n, dtype=np.float32)
    valid = np.zeros(n, dtype=np.float32)

    for start in range(0, n, TRAIN_BATCH):
        end = min(start + TRAIN_BATCH, n)
        h = np.asarray(train_h[start:end], dtype=np.complex64)
        power = leak.leaked_cell_power(leak.leaked_forward(h))
        total = power.sum(axis=(1, 2, 3, 4, 5), dtype=np.float64).astype(np.float32)
        denom = np.maximum(total, TINY)
        sh[start:end] = (power.sum(axis=(1, 4, 5), dtype=np.float64) / denom[:, None, None]).astype(np.float32)
        psh[start:end] = (power.sum(axis=(4, 5), dtype=np.float64) / denom[:, None, None, None]).astype(np.float32)
        shue[start:end] = (power.sum(axis=(1, 5), dtype=np.float64) / denom[:, None, None, None]).astype(np.float32)
        shv[start:end] = (power.sum(axis=(1, 4), dtype=np.float64) / denom[:, None, None, None]).astype(np.float32)
        total_power[start:end] = total
        valid[start:end] = (total > 1e-20).astype(np.float32)
        print(f"v209 train leaked marginals {end}/{n}", flush=True)

    np.savez(
        TRAIN_MARGINALS,
        sh=normalize(sh, axis=(1, 2)),
        psh=normalize(psh, axis=(1, 2, 3)),
        shue=normalize(shue, axis=(1, 2, 3)),
        shv=normalize(shv, axis=(1, 2, 3)),
        total_power=total_power,
        valid=valid,
    )
    data = np.load(TRAIN_MARGINALS)
    return {key: data[key].astype(np.float32) for key in data.files}


def weighted_train_target(train_arr: np.ndarray, idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.zeros((len(idx),) + train_arr.shape[1:], dtype=np.float32)
    reshape = (len(idx),) + (1,) * (train_arr.ndim - 1)
    for rank in range(idx.shape[1]):
        out += weights[:, rank].reshape(reshape).astype(np.float32) * train_arr[idx[:, rank]]
    return normalize(out, axis=tuple(range(1, out.ndim)))


def build_current_leaked_support() -> np.ndarray:
    if CURRENT_SUPPORT.is_file():
        return np.load(CURRENT_SUPPORT, mmap_mode="r")

    source = np.load(BASE, mmap_mode="r")
    out = np.lib.format.open_memmap(
        CURRENT_SUPPORT,
        mode="w+",
        dtype=np.float32,
        shape=(len(source), MP, S_SC, MH, N_ANT, MV),
    )
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h = np.asarray(source[start:end], dtype=np.complex64)
        power = leak.leaked_cell_power(leak.leaked_forward(h))
        out[start:end] = normalize(power, axis=(1, 2, 3, 4, 5))
        out.flush()
        print(f"v209 current leaked support {end}/{len(source)}", flush=True)
    return np.load(CURRENT_SUPPORT, mmap_mode="r")


def low_marginals_from_support(support: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "sh": normalize(np.asarray(support).sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32), axis=(1, 2)),
        "psh": normalize(np.asarray(support).sum(axis=(4, 5), dtype=np.float64).astype(np.float32), axis=(1, 2, 3)),
        "shue": normalize(np.asarray(support).sum(axis=(1, 5), dtype=np.float64).astype(np.float32), axis=(1, 2, 3)),
        "shv": normalize(np.asarray(support).sum(axis=(1, 4), dtype=np.float64).astype(np.float32), axis=(1, 2, 3)),
    }


def sparse_consensus(current: np.ndarray, matrix) -> np.ndarray:
    flat = current.reshape(len(current), -1)
    out = np.asarray(matrix @ flat, dtype=np.float32).reshape(current.shape)
    return normalize(out, axis=tuple(range(1, out.ndim)))


def l1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(a.reshape(len(a), -1) - b.reshape(len(b), -1)), axis=1).astype(np.float32)


def build_controller_and_target(support: np.ndarray, regimes: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if LEAKED_TARGET.is_file() and ACTION_OUT.is_file():
        target = np.load(LEAKED_TARGET, mmap_mode="r")
        actions = np.load(ACTION_OUT)
        if "alpha_sh" in actions.files:
            return target, {key: actions[key] for key in actions.files}

    require_file(V196_ACTIONS)
    require_file(V199_DIAG)
    require_file(V200_ACTIONS)

    train_marginals = build_train_leaked_marginals()
    v196_actions = np.load(V196_ACTIONS)
    v199_diag = np.load(V199_DIAG)
    v200_actions = np.load(V200_ACTIONS)

    train_idx = v196_actions["train_neighbor_idx"].astype(np.int32)
    train_w = v196_actions["train_neighbor_weights"].astype(np.float32)
    valid = train_marginals["valid"][train_idx].astype(np.float32)
    train_w = train_w * valid
    empty = train_w.sum(axis=1) <= TINY
    if np.any(empty):
        train_w[empty] = v196_actions["train_neighbor_weights"][empty].astype(np.float32)
    train_w /= np.maximum(train_w.sum(axis=1, keepdims=True), TINY)

    train_target = {
        "sh": weighted_train_target(train_marginals["sh"], train_idx, train_w),
        "psh": weighted_train_target(train_marginals["psh"], train_idx, train_w),
        "shue": weighted_train_target(train_marginals["shue"], train_idx, train_w),
        "shv": weighted_train_target(train_marginals["shv"], train_idx, train_w),
    }

    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    test_blocked = geom["test_blocked"].astype(np.float32)
    matrix, graph_diag = v189.build_weight_matrix(pos_test, test_blocked, regimes)

    current = low_marginals_from_support(support)
    test_target = {key: sparse_consensus(value, matrix) for key, value in current.items()}

    agree_sh = np.clip(1.0 - 0.50 * l1(train_target["sh"], test_target["sh"]), 0.0, 1.0)
    agree_psh = np.clip(1.0 - 0.35 * l1(train_target["psh"], test_target["psh"]), 0.0, 1.0)
    agree_shue = np.clip(1.0 - 0.28 * l1(train_target["shue"], test_target["shue"]), 0.0, 1.0)
    agree_shv = np.clip(1.0 - 0.24 * l1(train_target["shv"], test_target["shv"]), 0.0, 1.0)
    agree_mean = (0.34 * agree_sh + 0.24 * agree_psh + 0.22 * agree_shue + 0.20 * agree_shv).astype(np.float32)

    train_conf = v196_actions["train_confidence"].astype(np.float32)
    test_agreement = graph_diag["neighbor_agreement"].astype(np.float32)
    power_conf = v200_actions["power_confidence"].astype(np.float32)
    v196_gain_max = v196_actions["gain_max"].astype(np.float32)
    support_stress = np.clip((v196_gain_max - 4.0) / 2.8, 0.0, 1.0).astype(np.float32)
    power_active = v200_actions["active"].astype(bool)

    quality = (
        0.18
        + 0.24 * train_conf
        + 0.22 * test_agreement
        + 0.26 * agree_mean
        + 0.13 * power_conf
        - 0.10 * support_stress
    ).astype(np.float32)
    quality = np.clip(quality, 0.0, 1.22)

    alpha_base = np.asarray([LEAK_ALPHA_BASE[str(r)] for r in regimes], dtype=np.float32)
    alpha_cap = np.asarray([LEAK_ALPHA_CAP[str(r)] for r in regimes], dtype=np.float32)
    active_bonus = (power_active & (agree_mean > 0.55) & (power_conf > 0.66)).astype(np.float32)
    alpha_core = alpha_base * quality + 0.075 * alpha_cap * active_bonus
    alpha_core = np.minimum(alpha_core, alpha_cap).astype(np.float32)

    alpha_sh = alpha_core
    alpha_psh = np.minimum(0.78 * alpha_core, alpha_cap).astype(np.float32)
    alpha_shue = np.minimum(0.70 * alpha_core, alpha_cap).astype(np.float32)
    alpha_shv = np.minimum(0.66 * alpha_core, alpha_cap).astype(np.float32)

    train_mix = np.asarray([TRAIN_MIX_BASE[str(r)] for r in regimes], dtype=np.float32)
    train_mix = np.clip(train_mix + 0.16 * (agree_mean - 0.55), 0.12, 0.62).astype(np.float32)
    train_mix[regimes.astype(str) == "clear"] = 0.0

    desired = {}
    alpha_map = {"sh": alpha_sh, "psh": alpha_psh, "shue": alpha_shue, "shv": alpha_shv}
    for key in ["sh", "psh", "shue", "shv"]:
        shape = (len(regimes),) + (1,) * (current[key].ndim - 1)
        combined = (
            train_mix.reshape(shape) * train_target[key]
            + (1.0 - train_mix).reshape(shape) * test_target[key]
        )
        desired[key] = normalize(
            (1.0 - alpha_map[key].reshape(shape)) * current[key]
            + alpha_map[key].reshape(shape) * combined,
            axis=tuple(range(1, current[key].ndim)),
        )

    n = len(regimes)
    work = np.asarray(support, dtype=np.float32).copy()
    for _ in range(IPF_ITERS):
        apply_leaked_ratio(work, safe_ratio(desired["sh"], work.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)), "sh", 0.76)
        apply_leaked_ratio(work, safe_ratio(desired["psh"], work.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)), "psh", 0.44)
        apply_leaked_ratio(work, safe_ratio(desired["shue"], work.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)), "shue", 0.48)
        apply_leaked_ratio(work, safe_ratio(desired["shv"], work.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)), "shv", 0.42)
        work = normalize(work.reshape(n, -1), axis=1).reshape(support.shape)

    target = np.lib.format.open_memmap(LEAKED_TARGET, mode="w+", dtype=np.float32, shape=support.shape)
    target[:] = work.astype(np.float32)
    target.flush()

    target_low = low_marginals_from_support(work)
    diag = {
        "regimes": regimes.astype("U8"),
        "alpha_core": alpha_core,
        "alpha_sh": alpha_sh,
        "alpha_psh": alpha_psh,
        "alpha_shue": alpha_shue,
        "alpha_shv": alpha_shv,
        "controller_quality": quality,
        "train_mix": train_mix,
        "train_confidence": train_conf,
        "test_neighbor_agreement": test_agreement,
        "power_confidence": power_conf,
        "power_active": power_active,
        "support_stress": support_stress,
        "train_test_agree_sh": agree_sh,
        "train_test_agree_psh": agree_psh,
        "train_test_agree_shue": agree_shue,
        "train_test_agree_shv": agree_shv,
        "train_test_agree_mean": agree_mean,
        "target_full_l1": np.sum(np.abs(work.reshape(n, -1) - np.asarray(support).reshape(n, -1)), axis=1).astype(np.float32),
        "target_sh_l1": l1(target_low["sh"], current["sh"]),
        "target_psh_l1": l1(target_low["psh"], current["psh"]),
        "target_shue_l1": l1(target_low["shue"], current["shue"]),
        "target_shv_l1": l1(target_low["shv"], current["shv"]),
        "train_target_sh_l1": l1(train_target["sh"], current["sh"]),
        "test_target_sh_l1": l1(test_target["sh"], current["sh"]),
        "neighbor_idx": graph_diag["neighbor_idx"].astype(np.int32),
        "neighbor_weights": graph_diag["neighbor_weights"].astype(np.float32),
    }
    return np.load(LEAKED_TARGET, mmap_mode="r"), diag


def apply_leaked_ratio(work: np.ndarray, ratio: np.ndarray, axes: str, exponent: float) -> None:
    ratio = np.maximum(ratio, FLOOR).astype(np.float32) ** np.float32(exponent)
    if axes == "sh":
        work *= ratio[:, None, :, :, None, None]
    elif axes == "psh":
        work *= ratio[:, :, :, :, None, None]
    elif axes == "shue":
        work *= ratio[:, None, :, :, :, None]
    elif axes == "shv":
        work *= ratio[:, None, :, :, None, :]
    else:
        raise ValueError(axes)


def apply_leaked_projection(target: np.ndarray, regimes: np.ndarray) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(INTERMEDIATE, mode="w+", dtype=np.complex64, shape=source.shape)
    diag: dict[str, list[np.ndarray]] = {
        "leaked_actual_power_factor": [],
        "leaked_full_l1_actual": [],
        "leaked_sh_l1_actual": [],
        "leaked_gain_min": [],
        "leaked_gain_p10": [],
        "leaked_gain_p90": [],
        "leaked_gain_max": [],
    }

    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        spec0 = leak.leaked_forward(h0)
        power0 = leak.leaked_cell_power(spec0)
        total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        support0 = power0 / total0[:, None, None, None, None, None]
        h_power0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)

        target_power = target[start:end] * total0[:, None, None, None, None, None]
        gain = np.sqrt(target_power / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.16, 7.20).astype(np.float32)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.40, 2.95)
        gain[strong1024] = np.clip(gain[strong1024], 0.56, 2.05)
        gain[strong128] = np.clip(gain[strong128], 0.74, 1.54)

        for local, regime in enumerate(regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.48 * (gain[local] - 1.0)

        spec1 = spec0 * gain
        h1 = leak.leaked_inverse(spec1)
        h_power1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        h1 *= np.sqrt(h_power0 / h_power1)[:, None, None, None].astype(np.float32)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite leaked block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()

        power1 = leak.leaked_cell_power(leak.leaked_forward(h1.astype(np.complex64)))
        support1 = power1 / np.maximum(power1.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        diag["leaked_actual_power_factor"].append(
            (np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / h_power0).astype(np.float32)
        )
        diag["leaked_full_l1_actual"].append(np.sum(np.abs(support1 - support0), axis=(1, 2, 3, 4, 5)).astype(np.float32))
        diag["leaked_sh_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 4, 5), dtype=np.float64)
                    - support0.sum(axis=(1, 4, 5), dtype=np.float64)
                ),
                axis=(1, 2),
            ).astype(np.float32)
        )
        gflat = gain.reshape(end - start, -1)
        diag["leaked_gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["leaked_gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["leaked_gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["leaked_gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"v209 leaked projection {end}/{len(source)}", flush=True)

    return {key: np.concatenate(values).astype(np.float32) for key, values in diag.items()}


def build_delay_alpha(regimes: np.ndarray, actions: dict[str, np.ndarray]) -> np.ndarray:
    base = np.asarray([DELAY_EXTRA_BASE[str(r)] for r in regimes], dtype=np.float32)
    cap = np.asarray([DELAY_EXTRA_CAP[str(r)] for r in regimes], dtype=np.float32)
    quality = actions["controller_quality"].astype(np.float32)
    agree = actions["train_test_agree_mean"].astype(np.float32)
    alpha = base * np.clip(0.50 + 0.58 * quality + 0.22 * agree, 0.0, 1.25)
    alpha = np.minimum(alpha, cap).astype(np.float32)
    alpha[actions["support_stress"].astype(np.float32) > 0.85] *= 0.55
    return alpha.astype(np.float32)


def apply_delay_anchor_and_power(regimes: np.ndarray, actions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    source = np.load(INTERMEDIATE, mmap_mode="r")
    v196_target = np.load(V196_DELAY_TARGET, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    delay_alpha = build_delay_alpha(regimes, actions)
    extra_log_delta = build_power_extra(regimes, actions, delay_alpha)
    amp_extra = np.exp(0.5 * extra_log_delta).astype(np.float32)

    diag: dict[str, list[np.ndarray]] = {
        "delay_actual_power_factor": [],
        "delay_full_l1_actual": [],
        "delay_d_l1_actual": [],
        "final_actual_power_factor": [],
        "delay_gain_min": [],
        "delay_gain_p10": [],
        "delay_gain_p90": [],
        "delay_gain_max": [],
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

        local_alpha = delay_alpha[start:end]
        shape = (end - start,) + (1,) * (support0.ndim - 1)
        desired = normalize(
            (1.0 - local_alpha.reshape(shape)) * support0
            + local_alpha.reshape(shape) * np.asarray(v196_target[start:end], dtype=np.float32),
            axis=(1, 2, 3, 4, 5),
        )
        target_power = desired * total0[:, None, None, None, None, None]
        gain = np.sqrt(target_power / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.24, 4.20).astype(np.float32)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.50, 2.30)
        gain[strong1024] = np.clip(gain[strong1024], 0.64, 1.70)
        gain[strong128] = np.clip(gain[strong128], 0.80, 1.34)

        for local, regime in enumerate(regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.35 * (gain[local] - 1.0)

        arr1 = arr0 * gain
        h1 = np.fft.fft(arr1.reshape(end - start, 256, N_ANT, S_SC), axis=-1)
        h_power1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        h1 *= np.sqrt(h_power0 / h_power1)[:, None, None, None].astype(np.float32)
        h_delay_power = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)

        h2 = h1 * amp_extra[start:end, None, None, None].astype(np.float32)
        if not (np.isfinite(h2.real).all() and np.isfinite(h2.imag).all()):
            raise FloatingPointError(f"non-finite final block {start}:{end}")
        output[start:end] = h2.astype(np.complex64)
        output.flush()

        power1 = v196.delay_power_from_h(h1.astype(np.complex64))
        support1 = power1 / np.maximum(power1.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        diag["delay_actual_power_factor"].append((h_delay_power / h_power0).astype(np.float32))
        diag["final_actual_power_factor"].append(
            (
                np.sum(np.abs(h2) ** 2, axis=(1, 2, 3), dtype=np.float64)
                / np.maximum(h_power0, TINY)
            ).astype(np.float32)
        )
        diag["delay_full_l1_actual"].append(np.sum(np.abs(support1 - support0), axis=(1, 2, 3, 4, 5)).astype(np.float32))
        diag["delay_d_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 2, 3, 4), dtype=np.float64)
                    - support0.sum(axis=(1, 2, 3, 4), dtype=np.float64)
                ),
                axis=1,
            ).astype(np.float32)
        )
        gflat = gain.reshape(end - start, -1)
        diag["delay_gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["delay_gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["delay_gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["delay_gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"v209 delay/power projection {end}/{len(source)}", flush=True)

    out = {key: np.concatenate(values).astype(np.float32) for key, values in diag.items()}
    out["delay_alpha"] = delay_alpha
    out["extra_log_delta"] = extra_log_delta
    out["extra_power_factor"] = np.exp(extra_log_delta).astype(np.float32)
    return out


def build_power_extra(regimes: np.ndarray, actions: dict[str, np.ndarray], delay_alpha: np.ndarray) -> np.ndarray:
    v199_diag = np.load(V199_DIAG)
    v200_actions = np.load(V200_ACTIONS)
    labels = v199_diag["power_label"].astype(str)
    d_knn = v199_diag["delta_knn"].astype(np.float32)
    d_model = v199_diag["delta_model"].astype(np.float32)
    d_consensus = v199_diag["delta_consensus"].astype(np.float32)
    confidence = v199_diag["power_confidence"].astype(np.float32)
    model_spread = v199_diag["test_model_spread"].astype(np.float32)
    local_std = v199_diag["test_local_log_std"].astype(np.float32)
    v200_delta = v200_actions["final_log_delta"].astype(np.float32)

    train_model = (0.5 * (d_knn + d_model)).astype(np.float32)
    local_ratio = np.abs(d_consensus) / np.maximum(local_std, 1e-6)
    support_ready = (
        (actions["alpha_core"].astype(np.float32) > 0.18)
        & (actions["train_test_agree_mean"].astype(np.float32) > 0.48)
        & (delay_alpha > 0.055)
    )
    continue_mask = (
        support_ready
        & (np.abs(v200_delta) > 1e-7)
        & (np.sign(d_knn) == np.sign(d_model))
        & (confidence > 0.68)
    )
    conflict_mask = (
        support_ready
        & (labels == "conflict")
        & (np.sign(d_knn) == np.sign(d_model))
        & (np.abs(d_knn) > 0.42)
        & (np.abs(d_model) > 0.42)
        & (confidence > 0.68)
        & (local_ratio < 1.05)
        & (model_spread < 2.85)
    )

    extra = np.zeros(len(regimes), dtype=np.float32)
    quality = np.clip(0.45 + 0.45 * actions["controller_quality"].astype(np.float32) + 0.20 * confidence, 0.0, 1.20)
    for regime in ["clear", "light", "medium", "heavy"]:
        rmask = regimes.astype(str) == regime
        up_cap = POWER_EXTRA_UP_CAP[regime]
        down_cap = POWER_EXTRA_DOWN_CAP[regime]

        cont = rmask & continue_mask
        cont_mag = np.clip(
            0.010
            + 0.018 * np.minimum(np.abs(v200_delta), 0.14) / 0.14
            + 0.010 * np.minimum(np.abs(train_model), 2.5) / 2.5,
            0.0,
            0.055,
        )
        cont_val = (np.sign(v200_delta) * cont_mag * quality).astype(np.float32)
        extra[cont & (cont_val > 0)] += np.minimum(cont_val[cont & (cont_val > 0)], up_cap)
        extra[cont & (cont_val < 0)] += np.maximum(cont_val[cont & (cont_val < 0)], down_cap)

        conf = rmask & conflict_mask
        conf_mag = np.clip(
            0.012
            + 0.020 * np.clip(1.0 - local_ratio / 1.05, 0.0, 1.0)
            + 0.015 * np.minimum(np.abs(train_model), 2.8) / 2.8,
            0.0,
            0.050,
        )
        conf_val = (np.sign(train_model) * conf_mag * quality).astype(np.float32)
        extra[conf & (conf_val > 0)] += np.minimum(conf_val[conf & (conf_val > 0)], 0.72 * up_cap)
        extra[conf & (conf_val < 0)] += np.maximum(conf_val[conf & (conf_val < 0)], 0.72 * down_cap)

    return extra.astype(np.float32)


def label_action_summary(labels: np.ndarray, regimes: np.ndarray, active: np.ndarray) -> dict[str, object]:
    out: dict[str, object] = {}
    for label in ["power_up", "power_down", "conflict", "no_action"]:
        lmask = labels.astype(str) == label
        item: dict[str, object] = {
            "label_count": int(np.sum(lmask)),
            "active_count": int(np.sum(lmask & active)),
        }
        for regime in ["clear", "light", "medium", "heavy"]:
            rmask = regimes.astype(str) == regime
            item[f"{regime}_label_count"] = int(np.sum(lmask & rmask))
            item[f"{regime}_active_count"] = int(np.sum(lmask & rmask & active))
        out[label] = item
    return out


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for src in [
            PATHS.code / "generate_v209_adaptive_metric_projection_controller.py",
            PATHS.code / "generate_v200_selective_twosided_power_gate.py",
            PATHS.code / "analyze_v199_power_field_reliability.py",
            PATHS.code / "generate_v196_train_guided_delay_pdp_mouth.py",
            PATHS.code / "analyze_v195_scorer_mouth_reverse_inference.py",
            PATHS.code / "generate_v189_leaked_multimarginal_overdrive.py",
            PATHS.code / "generate_v187_leaked_pas_lifecycle_projection.py",
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
    require_file(BASE)
    require_file(V200_ACTIONS)
    require_file(V199_DIAG)
    require_file(V196_ACTIONS)
    require_file(V196_DELAY_TARGET)

    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geom["test_blocked"].astype(np.float32))
    support = build_current_leaked_support()
    target, target_diag = build_controller_and_target(support, regimes)
    leaked_diag = apply_leaked_projection(target, regimes)
    final_diag = apply_delay_anchor_and_power(regimes, target_diag)
    action_diag = {**target_diag, **leaked_diag, **final_diag}

    np.savez_compressed(ACTION_OUT, **action_diag)

    comparison_to_v200 = track.compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v196 = track.compare_output_to(OUTPUT, V196_BASE, regimes)
    comparison_to_v189 = track.compare_output_to(OUTPUT, V189_BASE, regimes)
    comparison_to_v190 = track.compare_output_to(OUTPUT, V190_BASE, regimes)
    if comparison_to_v200["shape"] != [500, 256, 4, 192] or comparison_to_v200["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v200['shape']} {comparison_to_v200['dtype']}")
    if not comparison_to_v200["finite"]:
        raise ValueError("v209 output contains non-finite values")

    v199_diag = np.load(V199_DIAG)
    labels = v199_diag["power_label"].astype(str)
    extra_active = np.abs(final_diag["extra_log_delta"]) > 1e-7
    active_support = target_diag["alpha_core"].astype(np.float32) > 1e-7

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The current v200 ceiling is caused by hand-set projection strength. "
            "Official train H can calibrate stable leaked-PAS marginals, while "
            "test-test consensus preserves the online-validated transductive "
            "structure.  Once support moves in a high-confidence direction, a "
            "small additional two-sided power release can be safer than the "
            "v202 scalar-only expansion."
        ),
        "new_information_used": {
            "official_train_h": "only normalized leaked low marginals SH/PSH/SHU/SHV are averaged; no full truth slice or complex residual is copied",
            "v189_v196_v200_chain": "uses the already validated leaked PAS, delay/PDP and selective power domains as the action space",
            "v155_failure_boundary": "absolute full PAS/PDP train-truth slice projection is avoided; train target is mixed with current test-test consensus and gated by agreement",
            "v208_oracle_boundary": "support/shape is corrected before any extra power release because power-only replacement can be negative when support/phase is wrong",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "ipf_iters": IPF_ITERS,
            "leak_alpha_base": LEAK_ALPHA_BASE,
            "leak_alpha_cap": LEAK_ALPHA_CAP,
            "train_mix_base": TRAIN_MIX_BASE,
            "delay_extra_base": DELAY_EXTRA_BASE,
            "delay_extra_cap": DELAY_EXTRA_CAP,
            "power_extra_up_cap_log": POWER_EXTRA_UP_CAP,
            "power_extra_down_cap_log": POWER_EXTRA_DOWN_CAP,
            "projection_order": ["leaked low-marginal projection", "v196 delay target anchor", "extra two-sided power release"],
            "phase_policy": "preserve v200/leaked/delay domain phase; no learned absolute phase regression",
            "zero_power_train_policy": "exclude zero-power train rows only from normalized marginal averaging weights",
        },
        "action_counts": {
            "support_active": int(np.sum(active_support)),
            "power_extra_active": int(np.sum(extra_active)),
            "power_label_counts": dict(Counter(labels.tolist())),
            "power_extra_by_label_and_regime": label_action_summary(labels, regimes, extra_active),
        },
        "diagnostics": {
            "overall": {
                **{
                    key: summarize(value)
                    for key, value in action_diag.items()
                    if key
                    not in {
                        "regimes",
                        "neighbor_idx",
                        "neighbor_weights",
                    }
                },
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "alpha_core": target_diag["alpha_core"],
                    "controller_quality": target_diag["controller_quality"],
                    "train_test_agree_mean": target_diag["train_test_agree_mean"],
                    "target_full_l1": target_diag["target_full_l1"],
                    "leaked_full_l1_actual": leaked_diag["leaked_full_l1_actual"],
                    "delay_alpha": final_diag["delay_alpha"],
                    "delay_full_l1_actual": final_diag["delay_full_l1_actual"],
                    "extra_log_delta": final_diag["extra_log_delta"],
                    "extra_power_factor": final_diag["extra_power_factor"],
                    "final_actual_power_factor": final_diag["final_actual_power_factor"],
                },
            ),
        },
        "comparison_to_v200": comparison_to_v200,
        "comparison_to_v196": comparison_to_v196,
        "comparison_to_v189": comparison_to_v189,
        "comparison_to_v190_failed_boundary": comparison_to_v190,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "intermediate": PATHS.relative(INTERMEDIATE),
            "train_leaked_marginals": PATHS.relative(TRAIN_MARGINALS),
            "current_support": PATHS.relative(CURRENT_SUPPORT),
            "target": PATHS.relative(LEAKED_TARGET),
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
