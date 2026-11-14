#!/usr/bin/env python3
"""Generate geometry-consistent train residual AD prior.

This is a follow-up to v58/v64.  Those versions selected residual components
only by neighbor residual energy.  Here every candidate component is softly
matched to LOS/specular path candidates using delay and BS AoD bins.  For
reflection paths with a consistent wall plane id, neighbor residual coefficients
are phase-aligned by the excess path length before IDW mixing.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path

import numpy as np

from build_component_path_assignments import circular_abs_delta, path_angle_bins
from build_map_features import BS_POS
from evaluate_local_ad_paths import MH, MP, MV, N_ANT, S_SC, inverse_ad
from evaluate_s1style_geom_blend import geometry_cache, select_geom_neighbors
from generate_train_residual_ad_prior import (
    AD_SHAPE,
    beta_subcarriers,
    flat_to_hvs,
    idw_weights,
    make_geom_cfg,
    radial_phase,
    regime_label,
)


REGIMES = ("clear", "light", "medium", "heavy")


def gain_for_regime(blocked: int, args) -> tuple[float, str]:
    label = regime_label(int(blocked))
    gains = {
        "clear": float(args.clear_gain),
        "light": float(args.light_gain),
        "medium": float(args.medium_gain),
        "heavy": float(args.heavy_gain),
    }
    return gains[label], label


def top_components(score: np.ndarray, count: int) -> np.ndarray:
    flat = np.asarray(score, dtype=np.float64).reshape(-1)
    k = min(int(count), flat.size)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    ids = np.argpartition(flat, -k)[-k:]
    ids = ids[np.argsort(flat[ids])[::-1]]
    return ids.astype(np.int64)


def same_plane_path_index(path_plane: np.ndarray, path_type: np.ndarray, q_type: int, q_plane: int) -> int:
    if q_type == 0:
        return 0
    mask = (path_type == q_type) & (path_plane == q_plane)
    ids = np.flatnonzero(mask)
    return int(ids[0]) if ids.size else -1


def path_match_for_components(component_ids: np.ndarray, paths: dict, row: int, args) -> tuple[np.ndarray, np.ndarray]:
    """Return soft geometry gate and best path index for flat AD components."""

    hbin, vbin, sbin = flat_to_hvs(component_ids)
    hbin = hbin.astype(np.float32)
    vbin = vbin.astype(np.float32)
    sbin = sbin.astype(np.float32)

    valid = np.asarray(paths["valid"][row], dtype=bool)
    delay = np.asarray(paths["delay_bin"][row], dtype=np.float32)
    tx_dir = np.asarray(paths["tx_dir"][row], dtype=np.float32)
    path_score = np.asarray(paths["score"][row], dtype=np.float32)
    path_h, path_v = path_angle_bins(tx_dir, args.steering_scale, args.h_sign, args.v_sign)
    path_h = np.asarray(path_h, dtype=np.float32)
    path_v = np.asarray(path_v, dtype=np.float32)

    de = np.abs(sbin[:, None] - delay[None, :])
    he = circular_abs_delta(hbin[:, None], path_h[None, :], MH)
    ve = circular_abs_delta(vbin[:, None], path_v[None, :], MV)
    dist2 = (
        (de / float(args.delay_sigma)) ** 2
        + (he / float(args.h_sigma)) ** 2
        + (ve / float(args.v_sigma)) ** 2
    )
    geom = np.exp(-0.5 * dist2).astype(np.float32)
    geom = np.where(valid[None, :] & np.isfinite(geom), geom, 0.0)

    # A weak prior toward high-quality reflection candidates, but do not let
    # this dominate the actual angle-delay consistency.
    ps = np.nan_to_num(path_score, nan=-10.0, neginf=-10.0, posinf=5.0).astype(np.float32)
    ps = ps - np.max(ps[valid]) if np.any(valid) else ps
    pweight = np.exp(np.clip(float(args.path_score_weight) * ps, -6.0, 6.0)).astype(np.float32)
    geom *= pweight[None, :]

    best = np.argmax(geom, axis=1).astype(np.int64)
    gate = geom[np.arange(len(component_ids)), best]
    gate = np.clip(gate, 0.0, 1.0).astype(np.float32)
    return gate, best


def select_geom_components(score: np.ndarray, paths: dict, row: int, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pool = top_components(score, args.component_pool)
    gate, best_path = path_match_for_components(pool, paths, row, args)
    gated_score = score.reshape(-1)[pool].astype(np.float64) * (
        float(args.gate_floor) + (1.0 - float(args.gate_floor)) * np.power(gate, float(args.gate_power))
    )
    take = min(int(args.components), pool.size)
    order = np.argsort(gated_score)[::-1][:take]
    return pool[order].astype(np.int64), gate[order].astype(np.float32), best_path[order].astype(np.int64)


def predict_residual(
    q_pos: np.ndarray,
    q_blocked: int,
    q_row: int,
    pos: np.ndarray,
    train_blocked: np.ndarray,
    dist_train: np.ndarray,
    dist_q: float,
    res_ad,
    res_power,
    train_paths: dict,
    test_paths: dict,
    args,
) -> tuple[np.ndarray, dict]:
    geom_cfg = make_geom_cfg(args, args.residual_k)
    idx, _raw_d, eff_d = select_geom_neighbors(pos, train_blocked, q_pos, q_blocked, None, geom_cfg)
    weights = idw_weights(eff_d, args.residual_power)
    neigh_power = np.asarray(res_power[idx])
    score = np.tensordot(weights.astype(np.float32), neigh_power, axes=(0, 0))
    comp_ids, gates, q_path_ids = select_geom_components(score, test_paths, q_row, args)
    hs, vs, ss = flat_to_hvs(comp_ids)

    pred_ad = np.zeros((1,) + AD_SHAPE, dtype=np.complex64)
    q_path_type = np.asarray(test_paths["path_type"][q_row])
    q_plane = np.asarray(test_paths["plane_id"][q_row])
    q_len = np.asarray(test_paths["path_length"][q_row], dtype=np.float64)
    q_excess = q_len - float(dist_q)

    aligned_count = 0
    usable_weight_sum = 0.0
    for local, (h, v, s, gate, qp) in enumerate(zip(hs, vs, ss, gates, q_path_ids)):
        qtype = int(q_path_type[int(qp)])
        qplane = int(q_plane[int(qp)])
        qextra = float(q_excess[int(qp)]) if np.isfinite(q_excess[int(qp)]) else 0.0
        coeff = np.zeros((MP, N_ANT), dtype=np.complex64)
        wsum = 0.0
        for w, ridx in zip(weights, idx):
            ridx = int(ridx)
            ww = float(w)
            if args.phase_align:
                tp = same_plane_path_index(
                    np.asarray(train_paths["plane_id"][ridx]),
                    np.asarray(train_paths["path_type"][ridx]),
                    qtype,
                    qplane,
                )
                if tp < 0 or not bool(train_paths["valid"][ridx, tp]):
                    if args.drop_unmatched_phase:
                        continue
                    phase = 1.0 + 0.0j
                else:
                    textra = float(train_paths["path_length"][ridx, tp] - dist_train[ridx])
                    phase = np.exp(-1j * float(args.beta) * (qextra - textra)).astype(np.complex64)
                    aligned_count += 1
            else:
                phase = 1.0 + 0.0j
            coeff += np.complex64(ww * phase) * np.asarray(res_ad[ridx, :, int(h), int(v), :, int(s)])
            wsum += ww
        if args.drop_unmatched_phase and wsum > 1e-12:
            coeff /= np.float32(wsum)
        coeff *= np.float32(float(args.gate_floor_coeff) + (1.0 - float(args.gate_floor_coeff)) * float(gate))
        pred_ad[0, :, int(h), int(v), :, int(s)] = coeff
        usable_weight_sum += wsum

    residual = inverse_ad(pred_ad)[0]
    beta_s = beta_subcarriers(args.beta, args.gamma)
    phase = radial_phase(float(dist_q), beta_s, -1.0)
    residual = (residual * phase[None, None, :]).astype(np.complex64)
    selected_score_sum = float(np.sum(score.reshape(-1)[comp_ids], dtype=np.float64)) if comp_ids.size else 0.0
    total_score_sum = float(np.sum(score, dtype=np.float64))
    return residual, {
        "selected_components": int(comp_ids.size),
        "selected_component_energy_frac": selected_score_sum / max(total_score_sum, 1e-30),
        "gate_mean": float(np.mean(gates)) if gates.size else 0.0,
        "gate_min": float(np.min(gates)) if gates.size else 0.0,
        "gate_max": float(np.max(gates)) if gates.size else 0.0,
        "aligned_count": int(aligned_count),
        "usable_weight_sum": float(usable_weight_sum),
    }


def load_paths(npz_path: str, split: str) -> dict:
    z = np.load(npz_path)
    return {
        "valid": z[f"{split}_valid"],
        "path_type": z[f"{split}_path_type"],
        "plane_id": z[f"{split}_plane_id"],
        "path_length": z[f"{split}_path_length"],
        "delay_bin": z[f"{split}_delay_bin"],
        "score": z[f"{split}_score"],
        "tx_dir": z[f"{split}_tx_dir"],
    }


def generate(args) -> dict:
    data = Path(args.data)
    pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    v36 = np.load(args.v36, mmap_mode="r")
    if v36.shape != (len(test_pos), 256, 4, 192):
        raise ValueError(f"bad v36 shape {v36.shape}")
    train_blocked, test_blocked = geometry_cache(args.data, args.cache, False)
    res_ad = np.load(args.residual_ad_cache, mmap_mode="r")
    res_power = np.load(args.residual_power_cache, mmap_mode="r")
    train_paths = load_paths(args.paths, "train")
    test_paths = load_paths(args.paths, "test")
    dist_train = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    dist_test = np.linalg.norm(test_pos - BS_POS[None, :], axis=1)

    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=v36.shape)
    stats = {
        "regime_counts": {r: 0 for r in REGIMES},
        "gain_sum": 0.0,
        "base_power_sum": 0.0,
        "residual_power_sum": 0.0,
        "pred_power_sum": 0.0,
        "selected_component_energy_frac_sum": 0.0,
        "gate_sum": 0.0,
        "aligned_count": 0,
        "usable_weight_sum": 0.0,
    }
    for i in range(len(test_pos)):
        base = np.asarray(v36[i], dtype=np.complex64)
        residual, meta = predict_residual(
            test_pos[i],
            int(test_blocked[i]),
            i,
            pos,
            train_blocked,
            dist_train,
            float(dist_test[i]),
            res_ad,
            res_power,
            train_paths,
            test_paths,
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
        stats["selected_component_energy_frac_sum"] += float(meta["selected_component_energy_frac"])
        stats["gate_sum"] += float(meta["gate_mean"])
        stats["aligned_count"] += int(meta["aligned_count"])
        stats["usable_weight_sum"] += float(meta["usable_weight_sum"])
        if (i + 1) % args.batch == 0 or i + 1 == len(test_pos):
            out.flush()
            print(f"generated {i + 1}/{len(test_pos)}", flush=True)

    stats["gain_mean"] = stats["gain_sum"] / max(len(test_pos), 1)
    stats["residual_to_base_power"] = stats["residual_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["pred_to_base_power"] = stats["pred_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["selected_component_energy_frac_mean"] = stats["selected_component_energy_frac_sum"] / max(len(test_pos), 1)
    stats["gate_mean"] = stats["gate_sum"] / max(len(test_pos), 1)
    payload = {
        "method": "geom_path_residual_prior",
        "version": str(args.version),
        "uses_external_s1": False,
        "formula": "v36 + gain(regime) * inverse_AD(geometry-gated and path-phase-aligned train residual prior)",
        "paths": args.paths,
        "prediction": {
            "components": int(args.components),
            "component_pool": int(args.component_pool),
            "residual_k": int(args.residual_k),
            "residual_power": float(args.residual_power),
            "phase_align": bool(args.phase_align),
            "drop_unmatched_phase": bool(args.drop_unmatched_phase),
            "gate_floor": float(args.gate_floor),
            "gate_floor_coeff": float(args.gate_floor_coeff),
            "gate_power": float(args.gate_power),
            "sigmas": {
                "delay": float(args.delay_sigma),
                "h": float(args.h_sigma),
                "v": float(args.v_sigma),
            },
            "gains": {
                "clear": float(args.clear_gain),
                "light": float(args.light_gain),
                "medium": float(args.medium_gain),
                "heavy": float(args.heavy_gain),
            },
        },
        "stats": stats,
        "output": args.output,
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
    parser.add_argument("--residual-power-cache", default=str(PATHS.artifacts / "shared_features/train_v36_loo_residual_derot_ad_power.npy"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.version_report_dir(65) / "geom_path_residual_prior_v65_generate.json"))
    parser.add_argument("--version", default="v65")
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--h-sign", type=int, default=-1)
    parser.add_argument("--v-sign", type=int, default=1)
    parser.add_argument("--residual-k", type=int, default=32)
    parser.add_argument("--residual-power", type=float, default=2.0)
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--component-pool", type=int, default=64)
    parser.add_argument("--delay-sigma", type=float, default=2.0)
    parser.add_argument("--h-sigma", type=float, default=2.5)
    parser.add_argument("--v-sigma", type=float, default=1.5)
    parser.add_argument("--path-score-weight", type=float, default=0.10)
    parser.add_argument("--gate-floor", type=float, default=0.12)
    parser.add_argument("--gate-floor-coeff", type=float, default=0.35)
    parser.add_argument("--gate-power", type=float, default=0.75)
    parser.add_argument("--phase-align", action="store_true", default=True)
    parser.add_argument("--no-phase-align", dest="phase_align", action="store_false")
    parser.add_argument("--drop-unmatched-phase", action="store_true")
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
