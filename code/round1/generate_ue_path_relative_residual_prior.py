#!/usr/bin/env python3
"""Generate a UE-aware path-relative residual prior.

This is a conservative extension of ``generate_path_relative_residual_prior``.
The v68 path-relative prior treats each residual component as

    (BS h-bin, BS v-bin, delay-bin) -> complex coefficients over (BS pol, UE antenna)

which keeps the UE antenna coefficients in the value but does not use UE angle
when assigning or transporting components.  This version splits the 4 UE
antennas into ``N_V=2`` vertical array bins and ``N_P=2`` polarizations, then
tracks residuals in

    (BS h-bin, BS v-bin, UE v-bin, delay-bin)

relative to geometric LOS/reflection candidates.

It is still not ray tracing output.  Geometry only defines candidate path
coordinates; the residual coefficients are learned from training-set LOO
errors and locally interpolated.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_component_path_assignments import circular_abs_delta, path_angle_bins
from build_map_features import BS_POS
from evaluate_local_ad_paths import MH, MP, MV, S_SC, inverse_ad
from evaluate_s1style_geom_blend import geometry_cache, select_geom_neighbors
from generate_train_residual_ad_prior import (
    beta_subcarriers,
    idw_weights,
    make_geom_cfg,
    radial_phase,
    regime_label,
)


REGIMES = ("clear", "light", "medium", "heavy")
NV, NP = 2, 2
N_ANT = NV * NP
UE_COMP_SHAPE = (MH, MV, NV, S_SC)


def signed_circular_delta(value: int, center: int, period: int) -> int:
    return int(((int(value) - int(center) + period // 2) % period) - period // 2)


def flat_to_hvus(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64)
    h = ids // (MV * NV * S_SC)
    rem = ids % (MV * NV * S_SC)
    v = rem // (NV * S_SC)
    rem = rem % (NV * S_SC)
    u = rem // S_SC
    s = rem % S_SC
    return h.astype(np.int64), v.astype(np.int64), u.astype(np.int64), s.astype(np.int64)


def top_flat_components(power: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(power, dtype=np.float64).reshape(-1)
    k = min(int(count), flat.size)
    if k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    ids = np.argpartition(flat, -k)[-k:]
    ids = ids[np.argsort(flat[ids])[::-1]]
    return ids.astype(np.int64), flat[ids].astype(np.float64)


def load_paths(npz_path: str) -> dict[str, dict[str, np.ndarray]]:
    z = np.load(npz_path)
    out: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "test"):
        out[split] = {
            "valid": np.asarray(z[f"{split}_valid"]),
            "path_type": np.asarray(z[f"{split}_path_type"]),
            "plane_id": np.asarray(z[f"{split}_plane_id"]),
            "path_length": np.asarray(z[f"{split}_path_length"], dtype=np.float32),
            "delay_bin": np.asarray(z[f"{split}_delay_bin"], dtype=np.float32),
            "score": np.asarray(z[f"{split}_score"], dtype=np.float32),
            "tx_dir": np.asarray(z[f"{split}_tx_dir"], dtype=np.float32),
            "rx_dir": np.asarray(z[f"{split}_rx_dir"], dtype=np.float32),
        }
    return out


def path_bins(paths: dict[str, np.ndarray], args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, v = path_angle_bins(paths["tx_dir"], args.steering_scale, args.h_sign, args.v_sign)
    return h.astype(np.float32), v.astype(np.float32), paths["delay_bin"].astype(np.float32)


def path_ue_bins(paths: dict[str, np.ndarray], args) -> np.ndarray:
    """Map last-hop RX direction to a coarse UE vertical DFT bin.

    With only two UE vertical elements this is intentionally coarse.  The
    learned relative offset table absorbs residual calibration error.
    """

    rz = np.asarray(paths["rx_dir"][..., 2], dtype=np.float32)
    u = (float(args.ue_sign) * float(args.ue_scale) * rz / (2.0 * np.pi) * NV + float(args.ue_offset)) % NV
    return u.astype(np.float32)


def coeff_to_ue_domain(coeff: np.ndarray, n_order: str) -> np.ndarray:
    """Convert (..., N_ANT) coefficients to (..., NV, NP) UE vertical spectrum."""

    coeff = np.asarray(coeff, dtype=np.complex64)
    if n_order == "vp":
        arr = coeff.reshape(coeff.shape[:-1] + (NV, NP))
    elif n_order == "pv":
        arr = coeff.reshape(coeff.shape[:-1] + (NP, NV)).swapaxes(-1, -2)
    else:
        raise ValueError(n_order)
    return np.fft.fft(arr, axis=-2).astype(np.complex64)


def coeff_from_ue_domain(ue_coeff: np.ndarray, n_order: str) -> np.ndarray:
    """Convert (..., NV, NP) UE vertical spectrum back to (..., N_ANT)."""

    arr = np.fft.ifft(np.asarray(ue_coeff, dtype=np.complex64), axis=-2).astype(np.complex64)
    if n_order == "vp":
        return arr.reshape(arr.shape[:-2] + (N_ANT,)).astype(np.complex64)
    if n_order == "pv":
        return arr.swapaxes(-1, -2).reshape(arr.shape[:-2] + (N_ANT,)).astype(np.complex64)
    raise ValueError(n_order)


def ue_power_from_ad_row(ad_row: np.ndarray, n_order: str) -> np.ndarray:
    """Return residual power over (BS h, BS v, UE v, delay)."""

    ad_row = np.asarray(ad_row, dtype=np.complex64)
    if n_order == "vp":
        arr = ad_row.reshape(MP, MH, MV, NV, NP, S_SC)
    elif n_order == "pv":
        arr = ad_row.reshape(MP, MH, MV, NP, NV, S_SC).transpose(0, 1, 2, 4, 3, 5)
    else:
        raise ValueError(n_order)
    ue = np.fft.fft(arr, axis=3).astype(np.complex64)
    return np.sum(np.abs(ue) ** 2, axis=(0, 4), dtype=np.float64).astype(np.float32)


def offset_component(
    path_h: float,
    path_v: float,
    path_u: float,
    path_s: float,
    offset: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    ds, dh, dv, du = offset
    h = (int(round(float(path_h))) + int(dh)) % MH
    v = (int(round(float(path_v))) + int(dv)) % MV
    u = (int(round(float(path_u))) + int(du)) % NV
    s = int(round(float(path_s))) + int(ds)
    if s < 0 or s >= S_SC:
        return None
    return h, v, u, s


def build_offset_table(res_ad, train_paths: dict[str, np.ndarray], args) -> tuple[dict[int, list[dict]], dict]:
    ph, pv, ps = path_bins(train_paths, args)
    pu = path_ue_bins(train_paths, args)
    valid = train_paths["valid"]
    ptype = train_paths["path_type"]
    accum: dict[tuple[int, int, int, int, int], float] = defaultdict(float)
    counts: dict[tuple[int, int, int, int, int], int] = defaultdict(int)
    assigned_power = 0.0
    top_power = 0.0
    total_power = 0.0

    for i in range(res_ad.shape[0]):
        pwr = ue_power_from_ad_row(res_ad[i], args.n_order)
        total_power += float(np.sum(pwr, dtype=np.float64))
        ids, vals = top_flat_components(pwr, args.calibration_top_components)
        if ids.size == 0:
            continue
        hs, vs, us, ss = flat_to_hvus(ids)
        for val, h, v, u, s in zip(vals, hs, vs, us, ss):
            top_power += float(val)
            de = np.abs(float(s) - ps[i])
            he = circular_abs_delta(float(h), ph[i], MH)
            ve = circular_abs_delta(float(v), pv[i], MV)
            ue = circular_abs_delta(float(u), pu[i], NV)
            score = (
                (de / float(args.assign_delay_sigma)) ** 2
                + (he / float(args.assign_h_sigma)) ** 2
                + (ve / float(args.assign_v_sigma)) ** 2
                + (ue / float(args.assign_u_sigma)) ** 2
            )
            score = np.where(valid[i], score, np.inf)
            p = int(np.argmin(score))
            if not np.isfinite(score[p]):
                continue
            if (
                de[p] > args.assign_delay_thresh
                or he[p] > args.assign_h_thresh
                or ve[p] > args.assign_v_thresh
                or ue[p] > args.assign_u_thresh
            ):
                continue
            h0 = int(round(float(ph[i, p]))) % MH
            v0 = int(round(float(pv[i, p]))) % MV
            u0 = int(round(float(pu[i, p]))) % NV
            ds = int(round(float(s) - float(ps[i, p])))
            dh = signed_circular_delta(int(h), h0, MH)
            dv = signed_circular_delta(int(v), v0, MV)
            du = signed_circular_delta(int(u), u0, NV)
            key = (int(ptype[i, p]), ds, dh, dv, du)
            accum[key] += float(val)
            counts[key] += 1
            assigned_power += float(val)

        if (i + 1) % args.calibration_log_every == 0 or i + 1 == res_ad.shape[0]:
            print(f"calibrated UE offsets {i + 1}/{res_ad.shape[0]}", flush=True)

    by_type: dict[int, list[dict]] = {0: [], 1: []}
    max_by_type = {0: 0.0, 1: 0.0}
    for (typ, _ds, _dh, _dv, _du), power in accum.items():
        max_by_type[typ] = max(max_by_type[typ], power)
    for (typ, ds, dh, dv, du), power in sorted(accum.items(), key=lambda kv: kv[1], reverse=True):
        if counts[(typ, ds, dh, dv, du)] < args.min_offset_count:
            continue
        norm = power / max(max_by_type.get(typ, 0.0), 1e-30)
        by_type[typ].append(
            {
                "path_type": int(typ),
                "offset": [int(ds), int(dh), int(dv), int(du)],
                "power": float(power),
                "relative_weight": float(norm),
                "count": int(counts[(typ, ds, dh, dv, du)]),
            }
        )
    for typ in by_type:
        by_type[typ] = by_type[typ][: int(args.offsets_per_type)]

    summary = {
        "n_order": str(args.n_order),
        "ue_scale": float(args.ue_scale),
        "ue_sign": int(args.ue_sign),
        "ue_offset": float(args.ue_offset),
        "calibration_top_components": int(args.calibration_top_components),
        "total_residual_power": float(total_power),
        "top_component_power_frac": float(top_power / max(total_power, 1e-30)),
        "assigned_top_component_power_frac": float(assigned_power / max(top_power, 1e-30)),
        "offsets_by_type": by_type,
    }
    return by_type, summary


def same_plane_path_index(paths: dict[str, np.ndarray], row: int, q_type: int, q_plane: int) -> int:
    if q_type == 0:
        return 0 if bool(paths["valid"][row, 0]) else -1
    mask = (
        (paths["path_type"][row] == q_type)
        & (paths["plane_id"][row] == q_plane)
        & paths["valid"][row]
    )
    ids = np.flatnonzero(mask)
    return int(ids[0]) if ids.size else -1


def best_same_type_path(paths: dict[str, np.ndarray], row: int, q_type: int) -> int:
    mask = (paths["path_type"][row] == q_type) & paths["valid"][row]
    ids = np.flatnonzero(mask)
    return int(ids[0]) if ids.size else -1


def best_path_bin_fallback(
    paths: dict[str, np.ndarray],
    bins: tuple[np.ndarray, np.ndarray, np.ndarray],
    ue_bins: np.ndarray,
    row: int,
    q_type: int,
    qh: float,
    qv: float,
    qu: float,
    qs: float,
    args,
) -> tuple[int, float]:
    mask = (paths["path_type"][row] == q_type) & paths["valid"][row]
    ids = np.flatnonzero(mask)
    if ids.size == 0:
        return -1, 0.0
    ph, pv, ps = bins
    de = np.abs(ps[row, ids].astype(np.float32) - float(qs))
    he = circular_abs_delta(ph[row, ids].astype(np.float32), float(qh), MH)
    ve = circular_abs_delta(pv[row, ids].astype(np.float32), float(qv), MV)
    ue = circular_abs_delta(ue_bins[row, ids].astype(np.float32), float(qu), NV)
    dist2 = (
        (de / float(args.fallback_delay_sigma)) ** 2
        + (he / float(args.fallback_h_sigma)) ** 2
        + (ve / float(args.fallback_v_sigma)) ** 2
        + (ue / float(args.fallback_u_sigma)) ** 2
    )
    j = int(np.argmin(dist2))
    sim = float(np.exp(-0.5 * float(dist2[j])))
    return int(ids[j]), sim


def path_score_weight(score: float, args) -> float:
    if not np.isfinite(score):
        return 0.0
    return float(np.exp(np.clip(args.path_score_weight * float(score), -4.0, 4.0)))


def candidate_coeff(
    q_path: int,
    offset_rec: dict,
    q_bins: tuple[np.ndarray, np.ndarray, np.ndarray],
    train_bins: tuple[np.ndarray, np.ndarray, np.ndarray],
    q_ue_bins: np.ndarray,
    train_ue_bins: np.ndarray,
    q_paths: dict[str, np.ndarray],
    train_paths: dict[str, np.ndarray],
    q_row: int,
    idx: np.ndarray,
    weights: np.ndarray,
    res_ad,
    args,
) -> dict | None:
    typ = int(offset_rec["path_type"])
    offset = tuple(int(x) for x in offset_rec["offset"])
    qh = q_bins[0][q_row, q_path]
    qv = q_bins[1][q_row, q_path]
    qs = q_bins[2][q_row, q_path]
    qu = q_ue_bins[q_row, q_path]
    q_comp = offset_component(qh, qv, qu, qs, offset)
    if q_comp is None:
        return None
    q_h, q_v, q_u, q_s = q_comp
    q_plane = int(q_paths["plane_id"][q_row, q_path])

    raw = np.zeros((MP, NP), dtype=np.complex64)
    raw_power = 0.0
    wsum = 0.0
    same_plane_weight = 0.0
    fallback_weight = 0.0
    used = 0
    for w, ridx in zip(weights, idx):
        ridx = int(ridx)
        tp = same_plane_path_index(train_paths, ridx, typ, q_plane)
        local_factor = 1.0
        if tp < 0 and args.fallback_same_type:
            if args.fallback_path_bin:
                tp, sim = best_path_bin_fallback(
                    train_paths,
                    train_bins,
                    train_ue_bins,
                    ridx,
                    typ,
                    qh,
                    qv,
                    qu,
                    qs,
                    args,
                )
                local_factor = float(args.fallback_weight) * float(sim)
            else:
                tp = best_same_type_path(train_paths, ridx, typ)
                local_factor = float(args.fallback_weight)
        if tp < 0:
            continue
        comp = offset_component(
            train_bins[0][ridx, tp],
            train_bins[1][ridx, tp],
            train_ue_bins[ridx, tp],
            train_bins[2][ridx, tp],
            offset,
        )
        if comp is None:
            continue
        th, tv, tu, ts = comp
        ww = float(w) * local_factor
        c = np.asarray(res_ad[ridx, :, th, tv, :, ts], dtype=np.complex64)
        cu = coeff_to_ue_domain(c, args.n_order)[:, int(tu), :]
        raw += np.complex64(ww) * cu
        raw_power += float(ww) * float(np.sum(np.abs(cu) ** 2, dtype=np.float64))
        wsum += float(ww)
        if local_factor >= 0.999:
            same_plane_weight += float(ww)
        else:
            fallback_weight += float(ww)
        used += 1

    if wsum < float(args.min_support_weight):
        return None
    coeff = (raw / np.float32(wsum)).astype(np.complex64)
    coeff_power = float(np.sum(np.abs(coeff) ** 2, dtype=np.float64))
    avg_power = raw_power / max(wsum, 1e-30)
    coherence = coeff_power / max(avg_power, 1e-30)
    coherence = float(np.clip(coherence, 0.0, 1.0))
    support_gate = min(1.0, wsum / max(float(args.support_norm), 1e-12))
    coh_gate = float(args.coherence_floor) + (1.0 - float(args.coherence_floor)) * (
        coherence ** float(args.coherence_power)
    )
    coeff = (coeff * np.float32((support_gate ** float(args.support_power)) * coh_gate)).astype(np.complex64)
    coeff_power = float(np.sum(np.abs(coeff) ** 2, dtype=np.float64))
    score = coeff_power
    score *= float(offset_rec["relative_weight"]) ** float(args.offset_weight_power)
    score *= path_score_weight(float(q_paths["score"][q_row, q_path]), args)
    score *= support_gate
    score *= coh_gate
    return {
        "h": int(q_h),
        "v": int(q_v),
        "u": int(q_u),
        "s": int(q_s),
        "coeff": coeff,
        "score": float(score),
        "coeff_power": float(coeff_power),
        "coherence": float(coherence),
        "support": float(wsum),
        "same_plane_weight": float(same_plane_weight),
        "fallback_weight": float(fallback_weight),
        "used_neighbors": int(used),
        "path_type": int(typ),
        "path_index": int(q_path),
        "offset": list(offset),
    }


def scatter_ue_component(pred_ad: np.ndarray, c: dict) -> None:
    tmp = np.zeros((MP, NV, NP), dtype=np.complex64)
    tmp[:, int(c["u"]), :] = np.asarray(c["coeff"], dtype=np.complex64)
    normal_coeff = coeff_from_ue_domain(tmp, "vp" if c.get("n_order") is None else c["n_order"])
    pred_ad[0, :, int(c["h"]), int(c["v"]), :, int(c["s"])] += normal_coeff


def predict_residual(
    q_pos: np.ndarray,
    q_blocked: int,
    q_row: int,
    pos: np.ndarray,
    train_blocked: np.ndarray,
    dist_q: float,
    res_ad,
    train_paths: dict[str, np.ndarray],
    test_paths: dict[str, np.ndarray],
    train_bins: tuple[np.ndarray, np.ndarray, np.ndarray],
    test_bins: tuple[np.ndarray, np.ndarray, np.ndarray],
    train_ue_bins: np.ndarray,
    test_ue_bins: np.ndarray,
    offsets_by_type: dict[int, list[dict]],
    args,
) -> tuple[np.ndarray, dict]:
    geom_cfg = make_geom_cfg(args, args.residual_k)
    idx, _raw_d, eff_d = select_geom_neighbors(pos, train_blocked, q_pos, q_blocked, None, geom_cfg)
    weights = idw_weights(eff_d, args.residual_power)

    candidates = []
    q_valid = test_paths["valid"][q_row]
    q_type = test_paths["path_type"][q_row]
    for p in range(q_valid.shape[0]):
        if not bool(q_valid[p]):
            continue
        typ = int(q_type[p])
        for offset_rec in offsets_by_type.get(typ, []):
            cand = candidate_coeff(
                p,
                offset_rec,
                test_bins,
                train_bins,
                test_ue_bins,
                train_ue_bins,
                test_paths,
                train_paths,
                q_row,
                idx,
                weights,
                res_ad,
                args,
            )
            if cand is not None:
                cand["n_order"] = str(args.n_order)
                candidates.append(cand)
    if not candidates:
        return np.zeros((256, 4, S_SC), dtype=np.complex64), {
            "candidate_count": 0,
            "selected_components": 0,
            "selected_score_frac": 0.0,
            "coherence_mean": 0.0,
            "support_mean": 0.0,
            "same_plane_weight_mean": 0.0,
            "fallback_weight_mean": 0.0,
            "reflection_selected_frac": 0.0,
        }

    candidates.sort(key=lambda c: c["score"], reverse=True)
    selected = candidates[: min(int(args.components), len(candidates))]
    total_score = sum(max(0.0, c["score"]) for c in candidates)
    selected_score = sum(max(0.0, c["score"]) for c in selected)
    pred_ad = np.zeros((1, MP, MH, MV, N_ANT, S_SC), dtype=np.complex64)
    for c in selected:
        scatter_ue_component(pred_ad, c)

    residual = inverse_ad(pred_ad)[0]
    beta_s = beta_subcarriers(args.beta, args.gamma)
    phase = radial_phase(dist_q, beta_s, -1.0)
    residual = (residual * phase[None, None, :]).astype(np.complex64)
    return residual, {
        "candidate_count": int(len(candidates)),
        "selected_components": int(len(selected)),
        "selected_score_frac": float(selected_score / max(total_score, 1e-30)),
        "coherence_mean": float(np.mean([c["coherence"] for c in selected])) if selected else 0.0,
        "support_mean": float(np.mean([c["support"] for c in selected])) if selected else 0.0,
        "same_plane_weight_mean": float(np.mean([c["same_plane_weight"] for c in selected])) if selected else 0.0,
        "fallback_weight_mean": float(np.mean([c["fallback_weight"] for c in selected])) if selected else 0.0,
        "reflection_selected_frac": float(np.mean([c["path_type"] == 1 for c in selected])) if selected else 0.0,
    }


def gain_for_regime(blocked: int, args) -> tuple[float, str]:
    label = regime_label(int(blocked))
    gains = {
        "clear": float(args.clear_gain),
        "light": float(args.light_gain),
        "medium": float(args.medium_gain),
        "heavy": float(args.heavy_gain),
    }
    return gains[label], label


def generate(args) -> dict:
    data = Path(args.data)
    pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    v36 = np.load(args.v36, mmap_mode="r")
    if v36.shape != (len(test_pos), 256, 4, 192):
        raise ValueError(f"bad v36 shape {v36.shape}")

    train_blocked, test_blocked = geometry_cache(args.data, args.cache, False)
    paths = load_paths(args.paths)
    train_bins = path_bins(paths["train"], args)
    test_bins = path_bins(paths["test"], args)
    train_ue_bins = path_ue_bins(paths["train"], args)
    test_ue_bins = path_ue_bins(paths["test"], args)
    res_ad = np.load(args.residual_ad_cache, mmap_mode="r")
    offsets_by_type, offset_summary = build_offset_table(res_ad, paths["train"], args)
    dist_test = np.linalg.norm(test_pos - BS_POS[None, :], axis=1)

    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=v36.shape)
    stats = {
        "regime_counts": {r: 0 for r in REGIMES},
        "gain_sum": 0.0,
        "base_power_sum": 0.0,
        "residual_power_sum": 0.0,
        "pred_power_sum": 0.0,
        "candidate_count_sum": 0.0,
        "selected_components_sum": 0.0,
        "selected_score_frac_sum": 0.0,
        "coherence_sum": 0.0,
        "support_sum": 0.0,
        "same_plane_weight_sum": 0.0,
        "fallback_weight_sum": 0.0,
        "reflection_selected_frac_sum": 0.0,
    }
    for i in range(len(test_pos)):
        base = np.asarray(v36[i], dtype=np.complex64)
        residual, meta = predict_residual(
            test_pos[i],
            int(test_blocked[i]),
            i,
            pos,
            train_blocked,
            float(dist_test[i]),
            res_ad,
            paths["train"],
            paths["test"],
            train_bins,
            test_bins,
            train_ue_bins,
            test_ue_bins,
            offsets_by_type,
            args,
        )
        gain, label = gain_for_regime(int(test_blocked[i]), args)
        pred = (base + np.float32(gain) * residual).astype(np.complex64)
        if not (np.isfinite(pred.real).all() and np.isfinite(pred.imag).all()):
            raise FloatingPointError(f"non-finite prediction at {i}")
        out[i] = pred

        stats["regime_counts"][label] += 1
        stats["gain_sum"] += float(gain)
        stats["base_power_sum"] += float(np.sum(np.abs(base) ** 2, dtype=np.float64))
        stats["residual_power_sum"] += float(np.sum(np.abs(residual) ** 2, dtype=np.float64))
        stats["pred_power_sum"] += float(np.sum(np.abs(pred) ** 2, dtype=np.float64))
        stats["candidate_count_sum"] += float(meta["candidate_count"])
        stats["selected_components_sum"] += float(meta["selected_components"])
        stats["selected_score_frac_sum"] += float(meta["selected_score_frac"])
        stats["coherence_sum"] += float(meta["coherence_mean"])
        stats["support_sum"] += float(meta["support_mean"])
        stats["same_plane_weight_sum"] += float(meta["same_plane_weight_mean"])
        stats["fallback_weight_sum"] += float(meta["fallback_weight_mean"])
        stats["reflection_selected_frac_sum"] += float(meta["reflection_selected_frac"])

        if (i + 1) % args.batch == 0 or i + 1 == len(test_pos):
            out.flush()
            print(f"generated {i + 1}/{len(test_pos)}", flush=True)

    n = max(len(test_pos), 1)
    stats["gain_mean"] = stats["gain_sum"] / n
    stats["residual_to_base_power"] = stats["residual_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["pred_to_base_power"] = stats["pred_power_sum"] / max(stats["base_power_sum"], 1e-30)
    for key in [
        "candidate_count",
        "selected_components",
        "selected_score_frac",
        "coherence",
        "support",
        "same_plane_weight",
        "fallback_weight",
        "reflection_selected_frac",
    ]:
        stats[f"{key}_mean"] = stats.pop(f"{key}_sum") / n

    payload = {
        "method": "ue_path_relative_residual_prior",
        "version": str(args.version),
        "uses_external_s1": False,
        "formula": "v36 + gain(regime) * inverse_AD(path-relative train residual coefficients with UE vertical bin)",
        "paths": args.paths,
        "v36": args.v36,
        "output": args.output,
        "offset_calibration": offset_summary,
        "prediction": {
            "components": int(args.components),
            "offsets_per_type": int(args.offsets_per_type),
            "calibration_top_components": int(args.calibration_top_components),
            "n_order": str(args.n_order),
            "ue_scale": float(args.ue_scale),
            "ue_sign": int(args.ue_sign),
            "ue_offset": float(args.ue_offset),
            "residual_k": int(args.residual_k),
            "residual_power": float(args.residual_power),
            "fallback_same_type": bool(args.fallback_same_type),
            "fallback_path_bin": bool(args.fallback_path_bin),
            "fallback_weight": float(args.fallback_weight),
            "support_norm": float(args.support_norm),
            "coherence_floor": float(args.coherence_floor),
            "coherence_power": float(args.coherence_power),
            "gains": {
                "clear": float(args.clear_gain),
                "light": float(args.light_gain),
                "medium": float(args.medium_gain),
                "heavy": float(args.heavy_gain),
            },
        },
        "stats": stats,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--paths", default=str(PATHS.artifacts / "shared_features/path_candidates.npz"))
    parser.add_argument("--v36", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v36.npy"))
    parser.add_argument("--residual-ad-cache", default=str(PATHS.artifacts / "shared_features/train_v36_loo_residual_derot_ad.npy"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.version_report_dir(79) / "ue_path_relative_residual_prior_v79_generate.json"))
    parser.add_argument("--version", default="v79")
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--h-sign", type=int, default=-1)
    parser.add_argument("--v-sign", type=int, default=1)
    parser.add_argument("--n-order", choices=("vp", "pv"), default="vp")
    parser.add_argument("--ue-scale", type=float, default=3.14159265359)
    parser.add_argument("--ue-sign", type=int, default=1)
    parser.add_argument("--ue-offset", type=float, default=0.0)

    parser.add_argument("--residual-k", type=int, default=32)
    parser.add_argument("--residual-power", type=float, default=2.0)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--offsets-per-type", type=int, default=16)
    parser.add_argument("--calibration-top-components", type=int, default=48)
    parser.add_argument("--calibration-log-every", type=int, default=250)
    parser.add_argument("--min-offset-count", type=int, default=20)
    parser.add_argument("--assign-delay-sigma", type=float, default=2.0)
    parser.add_argument("--assign-h-sigma", type=float, default=2.5)
    parser.add_argument("--assign-v-sigma", type=float, default=1.5)
    parser.add_argument("--assign-u-sigma", type=float, default=1.0)
    parser.add_argument("--assign-delay-thresh", type=float, default=4.0)
    parser.add_argument("--assign-h-thresh", type=float, default=5.0)
    parser.add_argument("--assign-v-thresh", type=float, default=3.0)
    parser.add_argument("--assign-u-thresh", type=float, default=1.5)
    parser.add_argument("--fallback-same-type", action="store_true")
    parser.add_argument("--fallback-path-bin", action="store_true")
    parser.add_argument("--fallback-weight", type=float, default=0.25)
    parser.add_argument("--fallback-delay-sigma", type=float, default=2.0)
    parser.add_argument("--fallback-h-sigma", type=float, default=2.5)
    parser.add_argument("--fallback-v-sigma", type=float, default=1.5)
    parser.add_argument("--fallback-u-sigma", type=float, default=1.0)
    parser.add_argument("--min-support-weight", type=float, default=1e-5)
    parser.add_argument("--support-norm", type=float, default=0.35)
    parser.add_argument("--support-power", type=float, default=0.5)
    parser.add_argument("--coherence-floor", type=float, default=0.20)
    parser.add_argument("--coherence-power", type=float, default=0.5)
    parser.add_argument("--offset-weight-power", type=float, default=0.5)
    parser.add_argument("--path-score-weight", type=float, default=0.05)

    parser.add_argument("--clear-gain", type=float, default=0.25)
    parser.add_argument("--light-gain", type=float, default=0.25)
    parser.add_argument("--medium-gain", type=float, default=0.75)
    parser.add_argument("--heavy-gain", type=float, default=1.50)
    parser.add_argument("--los-penalty", type=float, default=1.0)
    parser.add_argument("--occ-penalty", type=float, default=0.5)
    parser.add_argument("--r-penalty", type=float, default=0.0)
    parser.add_argument("--theta-penalty", type=float, default=0.0)
    parser.add_argument("--occ-scale", type=float, default=80.0)
    parser.add_argument("--r-scale", type=float, default=80.0)
    parser.add_argument("--theta-scale-deg", type=float, default=30.0)
    return parser.parse_args()


def main():
    generate(parse_args())


if __name__ == "__main__":
    main()
