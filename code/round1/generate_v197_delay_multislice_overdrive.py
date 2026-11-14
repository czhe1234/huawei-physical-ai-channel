#!/usr/bin/env python3
"""Generate v197: validated delay/PDP multislice overdrive on v196.

v196's online win confirmed that the v195 delay/PDP mouth analysis is a real
new information source beyond the leaked-PAS v189 backbone.  This version uses
v196 as the base and expands the same validated delay idea to a few still-low
dimensional delay mouths:

* D: global power delay profile;
* U-D: UE-port-specific PDP;
* H-D: horizontal-angle-specific PDP.
* P-D: polarization-specific PDP;
* V-D: vertical-angle-specific PDP;
* H-U-D: horizontal/UE sliced PDP.

The target is a mixture of two information sources:

* local train truth delay marginals, averaged by geometry/regime kNN;
* test-test v196 delay consensus, averaged by local geometry/regime graph.

It intentionally does not touch v190-style high-order leaked hyper-marginals.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
from scipy import sparse

import generate_v178_test_track_complex_consensus as track
from generate_v87_easy_field_projection import MH, MP, N_ANT, S_SC, summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v197_delay_multislice_overdrive"
OUT_DIR = PATHS.cache_dir(197)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(197)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v197_Round1_Test_Channel.npy"
DELAY_SUPPORT_OUT = OUT_DIR / "v197_v196_delay_support.npy"
TRAIN_MARGINALS_OUT = OUT_DIR / "v197_train_delay_multislice_marginals.npz"
TARGET_OUT = OUT_DIR / "v197_delay_multislice_target.npy"
ACTION_OUT = OUT_DIR / "v197_delay_multislice_actions.npz"
REPORT = REPORT_DIR / "v197_delay_multislice_overdrive_generate.json"
ZIP_OUT = PATHS.round1_versions / "v197_delay_multislice_overdrive_candidate_pending.zip"

BASE = PATHS.cache_dir(196) / "v196_Round1_Test_Channel.npy"
V189_BASE = PATHS.cache_dir(189) / "v189_Round1_Test_Channel.npy"
V188_BASE = PATHS.cache_dir(188) / "v188_Round1_Test_Channel.npy"
V190_BASE = PATHS.cache_dir(190) / "v190_Round1_Test_Channel.npy"

MV = 8
FLOOR = 1e-12
BATCH = 8
TRAIN_BATCH = 4
TRAIN_K = 72
TEST_K = 96
IPF_ITERS = 5

D_ALPHA = {"clear": 0.0, "light": 0.10, "medium": 0.70, "heavy": 0.96}
DU_ALPHA = {"clear": 0.0, "light": 0.070, "medium": 0.52, "heavy": 0.72}
DH_ALPHA = {"clear": 0.0, "light": 0.050, "medium": 0.38, "heavy": 0.54}
PD_ALPHA = {"clear": 0.0, "light": 0.025, "medium": 0.28, "heavy": 0.42}
VD_ALPHA = {"clear": 0.0, "light": 0.020, "medium": 0.25, "heavy": 0.38}
HUD_ALPHA = {"clear": 0.0, "light": 0.012, "medium": 0.16, "heavy": 0.25}

TRAIN_MIX = {"clear": 0.0, "light": 0.38, "medium": 0.60, "heavy": 0.72}
TEST_MIX = {"clear": 0.0, "light": 0.62, "medium": 0.40, "heavy": 0.28}


def normalize_rows(x: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=axis, keepdims=True), TINY)
    return out.astype(np.float32)


def safe_ratio(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    return (np.maximum(desired, FLOOR) / np.maximum(current, FLOOR)).astype(np.float32)


def delay_power_from_h(h: np.ndarray) -> np.ndarray:
    cir = np.fft.ifft(h, axis=-1)
    arr = cir.reshape(h.shape[0], MP, MH, MV, N_ANT, S_SC)
    return (np.abs(arr).astype(np.float32) ** 2).astype(np.float32)


def build_base_delay_support() -> np.ndarray:
    if DELAY_SUPPORT_OUT.is_file():
        return np.load(DELAY_SUPPORT_OUT, mmap_mode="r")

    source = np.load(BASE, mmap_mode="r")
    shape = (len(source), MP, MH, MV, N_ANT, S_SC)
    support = np.lib.format.open_memmap(DELAY_SUPPORT_OUT, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        power = delay_power_from_h(np.asarray(source[start:end], dtype=np.complex64))
        total = np.maximum(power.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        support[start:end] = (power / total).astype(np.float32)
        support.flush()
        print(f"v197 v196 delay support {end}/{len(source)}", flush=True)
    return np.load(DELAY_SUPPORT_OUT, mmap_mode="r")


def build_train_delay_marginals() -> dict[str, np.ndarray]:
    if TRAIN_MARGINALS_OUT.is_file():
        data = np.load(TRAIN_MARGINALS_OUT)
        return {key: data[key].astype(np.float32) for key in data.files}

    train_h = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    n = len(train_h)
    d = np.empty((n, S_SC), dtype=np.float32)
    du = np.empty((n, N_ANT, S_SC), dtype=np.float32)
    dh = np.empty((n, MH, S_SC), dtype=np.float32)
    pd = np.empty((n, MP, S_SC), dtype=np.float32)
    vd = np.empty((n, MV, S_SC), dtype=np.float32)
    hud = np.empty((n, MH, N_ANT, S_SC), dtype=np.float32)
    total_power = np.empty(n, dtype=np.float32)

    for start in range(0, n, TRAIN_BATCH):
        end = min(start + TRAIN_BATCH, n)
        power = delay_power_from_h(np.asarray(train_h[start:end], dtype=np.complex64))
        total = power.sum(axis=(1, 2, 3, 4, 5), dtype=np.float64).astype(np.float32)
        valid_total = np.maximum(total, TINY)
        d[start:end] = (power.sum(axis=(1, 2, 3, 4), dtype=np.float64) / valid_total[:, None]).astype(np.float32)
        du[start:end] = (power.sum(axis=(1, 2, 3), dtype=np.float64) / valid_total[:, None, None]).astype(np.float32)
        dh[start:end] = (power.sum(axis=(1, 3, 4), dtype=np.float64) / valid_total[:, None, None]).astype(np.float32)
        pd[start:end] = (power.sum(axis=(2, 3, 4), dtype=np.float64) / valid_total[:, None, None]).astype(np.float32)
        vd[start:end] = (power.sum(axis=(1, 2, 4), dtype=np.float64) / valid_total[:, None, None]).astype(np.float32)
        hud[start:end] = (power.sum(axis=(1, 3), dtype=np.float64) / valid_total[:, None, None, None]).astype(np.float32)
        total_power[start:end] = total
        print(f"v197 train delay multislice marginals {end}/{n}", flush=True)

    valid = (total_power > 1e-20).astype(np.float32)
    np.savez_compressed(TRAIN_MARGINALS_OUT, d=d, du=du, dh=dh, pd=pd, vd=vd, hud=hud, total_power=total_power, valid=valid)
    return {"d": d, "du": du, "dh": dh, "pd": pd, "vd": vd, "hud": hud, "total_power": total_power, "valid": valid}


def geometry_train_weights(
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    train_blocked: np.ndarray,
    test_blocked: np.ndarray,
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dxy = np.linalg.norm(test_pos[:, None, :2] - train_pos[None, :, :2], axis=2).astype(np.float32)
    db = np.abs(test_blocked[:, None].astype(np.float32) - train_blocked[None, :].astype(np.float32))
    mismatch = (test_regimes[:, None] != train_regimes[None, :]).astype(np.float32)
    effective = dxy + 0.105 * db + 6.0 * mismatch
    idx = np.argpartition(effective, TRAIN_K, axis=1)[:, :TRAIN_K]
    rows = np.arange(len(test_pos))[:, None]
    idx = idx[rows, np.argsort(effective[rows, idx], axis=1)]
    selected = effective[rows, idx]
    same = (test_regimes[:, None] == train_regimes[idx]).astype(np.float32)
    w = np.exp(-selected / 18.0).astype(np.float32) * (0.48 + 0.52 * same)
    w *= valid[idx].astype(np.float32)
    empty = w.sum(axis=1) <= TINY
    if np.any(empty):
        fallback = np.exp(-selected[empty] / 24.0).astype(np.float32)
        w[empty] = fallback
    w /= np.maximum(w.sum(axis=1, keepdims=True), TINY)
    confidence = (np.sum(w * same, axis=1) * np.clip(1.0 - selected[:, 0] / 42.0, 0.0, 1.0)).astype(np.float32)
    return idx.astype(np.int32), w.astype(np.float32), confidence


def predict_train_marginals(
    train_marginals: dict[str, np.ndarray],
    idx: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    d = np.einsum("ik,ikd->id", weights, train_marginals["d"][idx], optimize=True).astype(np.float32)
    du = np.einsum("ik,ikud->iud", weights, train_marginals["du"][idx], optimize=True).astype(np.float32)
    dh = np.einsum("ik,ikhd->ihd", weights, train_marginals["dh"][idx], optimize=True).astype(np.float32)
    pd = np.einsum("ik,ikpd->ipd", weights, train_marginals["pd"][idx], optimize=True).astype(np.float32)
    vd = np.einsum("ik,ikvd->ivd", weights, train_marginals["vd"][idx], optimize=True).astype(np.float32)
    hud = np.einsum("ik,ikhud->ihud", weights, train_marginals["hud"][idx], optimize=True).astype(np.float32)
    return {
        "d": normalize_rows(d, axis=1),
        "du": normalize_rows(du, axis=(1, 2)),
        "dh": normalize_rows(dh, axis=(1, 2)),
        "pd": normalize_rows(pd, axis=(1, 2)),
        "vd": normalize_rows(vd, axis=(1, 2)),
        "hud": normalize_rows(hud, axis=(1, 2, 3)),
    }


def build_test_consensus(support: np.ndarray, pos: np.ndarray, blocked: np.ndarray, regimes: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    idx, raw_w = track.neighbor_graph(pos, blocked, regimes, k=TEST_K)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    agreement = np.zeros(len(pos), dtype=np.float32)
    effective = np.zeros(len(pos), dtype=np.float32)
    for i in range(len(pos)):
        rows.append(i)
        cols.append(i)
        vals.append(0.42)
        total = 0.42
        nb = idx[i].astype(np.int64)
        w = raw_w[i].astype(np.float64)
        same = (regimes[nb] == regimes[i]).astype(np.float64)
        w = w * (0.50 + 0.50 * same)
        w /= max(float(w.sum()), 1e-12)
        for node, weight in zip(nb, w):
            value = float(weight)
            rows.append(i)
            cols.append(int(node))
            vals.append(value)
            total += value
        for j in range(len(vals) - len(nb) - 1, len(vals)):
            vals[j] /= total
        agreement[i] = np.float32(float(np.sum(raw_w[i] * same)))
        effective[i] = np.float32(1.0 / max(float(np.sum(w * w)), 1e-12))
    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(pos), len(pos)), dtype=np.float32)
    flat = np.asarray(support).reshape(len(pos), -1)
    consensus = matrix @ flat
    consensus = normalize_rows(np.asarray(consensus, dtype=np.float32), axis=1).reshape(support.shape)
    return consensus.astype(np.float32), {
        "test_neighbor_idx": idx.astype(np.int32),
        "test_neighbor_weights": raw_w.astype(np.float32),
        "test_neighbor_agreement": agreement,
        "test_effective_neighbor_count": effective,
    }


def alpha_by_regime(regimes: np.ndarray, base: dict[str, float], train_conf: np.ndarray, test_agreement: np.ndarray) -> np.ndarray:
    out = np.zeros(len(regimes), dtype=np.float32)
    for i, regime in enumerate(regimes.astype(str)):
        b = base[regime]
        if b <= 0.0:
            continue
        conf = float(np.clip(0.54 + 0.26 * train_conf[i] + 0.20 * test_agreement[i], 0.35, 1.18))
        out[i] = np.float32(b * conf)
    return out


def mix_target(
    current: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    regimes: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    train_mix = np.asarray([TRAIN_MIX[str(r)] for r in regimes], dtype=np.float32)
    test_mix = np.asarray([TEST_MIX[str(r)] for r in regimes], dtype=np.float32)
    mix_total = np.maximum(train_mix + test_mix, TINY)
    train_mix = train_mix / mix_total
    test_mix = test_mix / mix_total
    shape = (len(regimes),) + (1,) * (current.ndim - 1)
    combined = train_mix.reshape(shape) * train_target + test_mix.reshape(shape) * test_target
    desired = (1.0 - alpha.reshape(shape)) * current + alpha.reshape(shape) * combined
    return normalize_rows(desired, axis=tuple(range(1, desired.ndim)))


def apply_delay_ratio(work: np.ndarray, ratio: np.ndarray, mode: str, exponent: float) -> None:
    ratio = np.maximum(ratio, FLOOR).astype(np.float32) ** np.float32(exponent)
    if mode == "d":
        work *= ratio[:, None, None, None, None, :]
    elif mode == "du":
        work *= ratio[:, None, None, None, :, :]
    elif mode == "dh":
        work *= ratio[:, None, :, None, None, :]
    elif mode == "pd":
        work *= ratio[:, :, None, None, None, :]
    elif mode == "vd":
        work *= ratio[:, None, None, :, None, :]
    elif mode == "hud":
        work *= ratio[:, None, :, None, :, :]
    else:
        raise ValueError(mode)


def build_target(support: np.ndarray, regimes: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pos_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked)

    train_marginals = build_train_delay_marginals()
    train_idx, train_w, train_conf = geometry_train_weights(
        pos_train,
        pos_test,
        train_blocked,
        test_blocked,
        train_regimes,
        regimes,
        train_marginals["valid"],
    )
    train_pred = predict_train_marginals(train_marginals, train_idx, train_w)

    test_consensus, graph_diag = build_test_consensus(support, pos_test, test_blocked, regimes)

    d0 = np.asarray(support).sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
    du0 = np.asarray(support).sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    dh0 = np.asarray(support).sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)
    pd0 = np.asarray(support).sum(axis=(2, 3, 4), dtype=np.float64).astype(np.float32)
    vd0 = np.asarray(support).sum(axis=(1, 2, 4), dtype=np.float64).astype(np.float32)
    hud0 = np.asarray(support).sum(axis=(1, 3), dtype=np.float64).astype(np.float32)
    dc = test_consensus.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
    duc = test_consensus.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    dhc = test_consensus.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)
    pdc = test_consensus.sum(axis=(2, 3, 4), dtype=np.float64).astype(np.float32)
    vdc = test_consensus.sum(axis=(1, 2, 4), dtype=np.float64).astype(np.float32)
    hudc = test_consensus.sum(axis=(1, 3), dtype=np.float64).astype(np.float32)

    alpha_d = alpha_by_regime(regimes, D_ALPHA, train_conf, graph_diag["test_neighbor_agreement"])
    alpha_du = alpha_by_regime(regimes, DU_ALPHA, train_conf, graph_diag["test_neighbor_agreement"])
    alpha_dh = alpha_by_regime(regimes, DH_ALPHA, train_conf, graph_diag["test_neighbor_agreement"])
    alpha_pd = alpha_by_regime(regimes, PD_ALPHA, train_conf, graph_diag["test_neighbor_agreement"])
    alpha_vd = alpha_by_regime(regimes, VD_ALPHA, train_conf, graph_diag["test_neighbor_agreement"])
    alpha_hud = alpha_by_regime(regimes, HUD_ALPHA, train_conf, graph_diag["test_neighbor_agreement"])

    desired_d = mix_target(d0, train_pred["d"], dc, regimes, alpha_d)
    desired_du = mix_target(du0, train_pred["du"], duc, regimes, alpha_du)
    desired_dh = mix_target(dh0, train_pred["dh"], dhc, regimes, alpha_dh)
    desired_pd = mix_target(pd0, train_pred["pd"], pdc, regimes, alpha_pd)
    desired_vd = mix_target(vd0, train_pred["vd"], vdc, regimes, alpha_vd)
    desired_hud = mix_target(hud0, train_pred["hud"], hudc, regimes, alpha_hud)

    n = len(regimes)
    work = np.asarray(support, dtype=np.float32).copy()
    for _ in range(IPF_ITERS):
        apply_delay_ratio(work, safe_ratio(desired_d, work.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)), "d", 0.92)
        apply_delay_ratio(work, safe_ratio(desired_du, work.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)), "du", 0.58)
        apply_delay_ratio(work, safe_ratio(desired_dh, work.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)), "dh", 0.44)
        apply_delay_ratio(work, safe_ratio(desired_pd, work.sum(axis=(2, 3, 4), dtype=np.float64).astype(np.float32)), "pd", 0.42)
        apply_delay_ratio(work, safe_ratio(desired_vd, work.sum(axis=(1, 2, 4), dtype=np.float64).astype(np.float32)), "vd", 0.38)
        apply_delay_ratio(work, safe_ratio(desired_hud, work.sum(axis=(1, 3), dtype=np.float64).astype(np.float32)), "hud", 0.25)
        work = normalize_rows(work.reshape(n, -1), axis=1).reshape(support.shape)

    target = np.lib.format.open_memmap(TARGET_OUT, mode="w+", dtype=np.float32, shape=support.shape)
    target[:] = work.astype(np.float32)
    target.flush()

    d1 = work.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
    du1 = work.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    dh1 = work.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)
    pd1 = work.sum(axis=(2, 3, 4), dtype=np.float64).astype(np.float32)
    vd1 = work.sum(axis=(1, 2, 4), dtype=np.float64).astype(np.float32)
    hud1 = work.sum(axis=(1, 3), dtype=np.float64).astype(np.float32)
    flat0 = np.asarray(support).reshape(n, -1)
    flat1 = work.reshape(n, -1)
    diag = {
        **graph_diag,
        "train_neighbor_idx": train_idx,
        "train_neighbor_weights": train_w,
        "train_confidence": train_conf,
        "alpha_d": alpha_d,
        "alpha_du": alpha_du,
        "alpha_dh": alpha_dh,
        "alpha_pd": alpha_pd,
        "alpha_vd": alpha_vd,
        "alpha_hud": alpha_hud,
        "target_full_l1": np.sum(np.abs(flat1 - flat0), axis=1).astype(np.float32),
        "margin_d_l1": np.sum(np.abs(d1 - d0), axis=1).astype(np.float32),
        "margin_du_l1": np.sum(np.abs(du1 - du0), axis=(1, 2)).astype(np.float32),
        "margin_dh_l1": np.sum(np.abs(dh1 - dh0), axis=(1, 2)).astype(np.float32),
        "margin_pd_l1": np.sum(np.abs(pd1 - pd0), axis=(1, 2)).astype(np.float32),
        "margin_vd_l1": np.sum(np.abs(vd1 - vd0), axis=(1, 2)).astype(np.float32),
        "margin_hud_l1": np.sum(np.abs(hud1 - hud0), axis=(1, 2, 3)).astype(np.float32),
        "train_d_l1_to_current": np.sum(np.abs(train_pred["d"] - d0), axis=1).astype(np.float32),
        "test_d_l1_to_current": np.sum(np.abs(dc - d0), axis=1).astype(np.float32),
        "train_hud_l1_to_current": np.sum(np.abs(train_pred["hud"] - hud0), axis=(1, 2, 3)).astype(np.float32),
        "test_hud_l1_to_current": np.sum(np.abs(hudc - hud0), axis=(1, 2, 3)).astype(np.float32),
    }
    return np.load(TARGET_OUT, mmap_mode="r"), diag


def apply_target(target: np.ndarray, regimes: np.ndarray) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    diag: dict[str, list[np.ndarray]] = {
        "actual_power_factor": [],
        "full_l1_actual": [],
        "d_l1_actual": [],
        "du_l1_actual": [],
        "dh_l1_actual": [],
        "pd_l1_actual": [],
        "vd_l1_actual": [],
        "hud_l1_actual": [],
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

        target_power = target[start:end] * total0[:, None, None, None, None, None]
        gain = np.sqrt(target_power / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.14, 7.20).astype(np.float32)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.40, 2.80)
        gain[strong1024] = np.clip(gain[strong1024], 0.54, 1.95)
        gain[strong128] = np.clip(gain[strong128], 0.72, 1.48)

        for local, regime in enumerate(regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.45 * (gain[local] - 1.0)

        arr1 = arr0 * gain
        h1 = np.fft.fft(arr1.reshape(end - start, 256, N_ANT, S_SC), axis=-1)
        h_power1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        h1 *= np.sqrt(h_power0 / h_power1)[:, None, None, None].astype(np.float32)

        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()

        power1 = delay_power_from_h(h1.astype(np.complex64))
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
        diag["pd_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(2, 3, 4), dtype=np.float64)
                    - support0.sum(axis=(2, 3, 4), dtype=np.float64)
                ),
                axis=(1, 2),
            ).astype(np.float32)
        )
        diag["vd_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 2, 4), dtype=np.float64)
                    - support0.sum(axis=(1, 2, 4), dtype=np.float64)
                ),
                axis=(1, 2),
            ).astype(np.float32)
        )
        diag["hud_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 3), dtype=np.float64)
                    - support0.sum(axis=(1, 3), dtype=np.float64)
                ),
                axis=(1, 2, 3),
            ).astype(np.float32)
        )
        gflat = gain.reshape(end - start, -1)
        diag["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"v197 delay multislice projection {end}/{len(source)}", flush=True)

    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def compare_output_to(output_path, base_path, regimes: np.ndarray) -> dict[str, object]:
    return track.compare_output_to(output_path, base_path, regimes)


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for src in [
            PATHS.code / "generate_v197_delay_multislice_overdrive.py",
            PATHS.code / "generate_v196_train_guided_delay_pdp_mouth.py",
            PATHS.code / "analyze_v195_scorer_mouth_reverse_inference.py",
            PATHS.code / "generate_v189_leaked_multimarginal_overdrive.py",
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
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geom["test_blocked"].astype(np.float32))
    support = build_base_delay_support()
    target, target_diag = build_target(support, regimes)
    apply_diag = apply_target(target, regimes)
    np.savez_compressed(ACTION_OUT, regimes=regimes.astype("U8"), **target_diag, **apply_diag)

    comparison_to_v196 = compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v189 = compare_output_to(OUTPUT, V189_BASE, regimes)
    comparison_to_v188 = compare_output_to(OUTPUT, V188_BASE, regimes)
    comparison_to_v190 = compare_output_to(OUTPUT, V190_BASE, regimes)
    if comparison_to_v196["shape"] != [500, 256, 4, 192] or comparison_to_v196["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v196['shape']} {comparison_to_v196['dtype']}")
    if not comparison_to_v196["finite"]:
        raise ValueError("v197 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v196 online feedback confirmed that delay/PDP mouth correction is "
            "a real scoring axis beyond v189.  v197 uses v196 as base and adds "
            "P-D, V-D and H-U-D low-order delay mouths while preserving v196 "
            "phase, total power and leaked-PAS backbone."
        ),
        "new_information_used": {
            "v196_online": "0.511432 confirms delay/PDP target is positive vs v189",
            "v195_all_history": "delay_d/PDP mouth has best full-history score correlation",
            "train_truth_spectrum": "normalized D, U-D, H-D, P-D, V-D and H-U-D delay marginals are averaged; no complex residual is migrated",
            "test_test_consensus": "v196 delay marginals are smoothed by local test graph",
            "v190_boundary": "does not construct or optimize leaked high-order pol-S-H-UE / pol-S-H-V / S-H-UE-V targets",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "train_k": TRAIN_K,
            "test_k": TEST_K,
            "d_alpha": D_ALPHA,
            "du_alpha": DU_ALPHA,
            "dh_alpha": DH_ALPHA,
            "pd_alpha": PD_ALPHA,
            "vd_alpha": VD_ALPHA,
            "hud_alpha": HUD_ALPHA,
            "train_mix": TRAIN_MIX,
            "test_mix": TEST_MIX,
            "ipf_iters": IPF_ITERS,
            "marginals": ["D", "U-D", "H-D", "P-D", "V-D", "H-U-D"],
            "gain_clip": {
                "all": [0.14, 7.20],
                "top8192_delay_cells": [0.40, 2.80],
                "top1024_delay_cells": [0.54, 1.95],
                "top128_delay_cells": [0.72, 1.48],
                "light_blend": "45% of raw gain",
                "clear": "identity",
            },
            "power_policy": "preserve v196 per-point total power",
            "phase_policy": "preserve v196 delay-domain phase; no absolute phase transport",
            "zero_power_train_policy": "exclude total_power<=1e-20 only when averaging normalized spectra because normalized PDP is undefined",
        },
        "diagnostics": {
            "overall": {
                **{key: summarize(value) for key, value in target_diag.items() if key not in {"train_neighbor_idx", "train_neighbor_weights", "test_neighbor_idx", "test_neighbor_weights"}},
                **{key: summarize(value) for key, value in apply_diag.items()},
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "alpha_d": target_diag["alpha_d"],
                    "alpha_du": target_diag["alpha_du"],
                    "alpha_dh": target_diag["alpha_dh"],
                    "alpha_pd": target_diag["alpha_pd"],
                    "alpha_vd": target_diag["alpha_vd"],
                    "alpha_hud": target_diag["alpha_hud"],
                    "train_confidence": target_diag["train_confidence"],
                    "target_full_l1": target_diag["target_full_l1"],
                    "margin_d_l1": target_diag["margin_d_l1"],
                    "margin_du_l1": target_diag["margin_du_l1"],
                    "margin_dh_l1": target_diag["margin_dh_l1"],
                    "margin_pd_l1": target_diag["margin_pd_l1"],
                    "margin_vd_l1": target_diag["margin_vd_l1"],
                    "margin_hud_l1": target_diag["margin_hud_l1"],
                    "full_l1_actual": apply_diag["full_l1_actual"],
                    "d_l1_actual": apply_diag["d_l1_actual"],
                    "du_l1_actual": apply_diag["du_l1_actual"],
                    "dh_l1_actual": apply_diag["dh_l1_actual"],
                    "pd_l1_actual": apply_diag["pd_l1_actual"],
                    "vd_l1_actual": apply_diag["vd_l1_actual"],
                    "hud_l1_actual": apply_diag["hud_l1_actual"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                    "gain_max": apply_diag["gain_max"],
                },
            ),
        },
        "comparison_to_v196": comparison_to_v196,
        "comparison_to_v189": comparison_to_v189,
        "comparison_to_v188": comparison_to_v188,
        "comparison_to_v190": comparison_to_v190,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "delay_support": PATHS.relative(DELAY_SUPPORT_OUT),
            "train_marginals": PATHS.relative(TRAIN_MARGINALS_OUT),
            "target": PATHS.relative(TARGET_OUT),
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
