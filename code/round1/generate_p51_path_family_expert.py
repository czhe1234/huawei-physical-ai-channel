#!/usr/bin/env python3
"""Generate a P51 path-family residual expert candidate.

v68 learns path-relative residual coefficients after pooling all reflection
planes into one offset table.  The path-family visualization showed that plane
51 dominates several strong, spatially coherent component families.  This
candidate keeps the v68 backbone, but replaces the generic reflection offset
table with a plane-specific expert whenever the query path is plane 51.

This is still an AI/data-driven residual model: geometry only defines the
coordinate system and candidate path identity; complex coefficients are learned
from the training residual cache.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_component_path_assignments import circular_abs_delta
from build_map_features import BS_POS
from evaluate_local_ad_paths import MH, MV, S_SC, inverse_ad
from evaluate_s1style_geom_blend import geometry_cache, select_geom_neighbors
from generate_path_relative_residual_prior import (
    build_offset_table,
    candidate_coeff,
    load_paths,
    path_bins,
    signed_circular_delta,
    top_flat_components,
)
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


def build_plane_offset_table(res_power, train_paths: dict[str, np.ndarray], args) -> tuple[list[dict], dict]:
    """Infer residual offsets specifically for one reflection plane."""

    ph, pv, ps = path_bins(train_paths, args)
    valid = train_paths["valid"]
    ptype = train_paths["path_type"]
    plane = train_paths["plane_id"]
    accum: dict[tuple[int, int, int], float] = defaultdict(float)
    counts: dict[tuple[int, int, int], int] = defaultdict(int)
    assigned_power = 0.0
    top_power = 0.0

    for i in range(res_power.shape[0]):
        ids, vals = top_flat_components(res_power[i], args.special_calibration_top_components)
        if ids.size == 0:
            continue
        hs, vs, ss = flat_to_hvs(ids)
        path_mask = (
            valid[i]
            & (ptype[i] == 1)
            & (plane[i] == int(args.special_plane_id))
        )
        pids = np.flatnonzero(path_mask)
        for val, h, v, s in zip(vals, hs, vs, ss):
            top_power += float(val)
            if pids.size == 0:
                continue
            de = np.abs(ps[i, pids].astype(np.float32) - float(s))
            he = circular_abs_delta(ph[i, pids].astype(np.float32), float(h), MH)
            ve = circular_abs_delta(pv[i, pids].astype(np.float32), float(v), MV)
            score = (
                (de / float(args.assign_delay_sigma)) ** 2
                + (he / float(args.assign_h_sigma)) ** 2
                + (ve / float(args.assign_v_sigma)) ** 2
            )
            j = int(np.argmin(score))
            p = int(pids[j])
            if de[j] > args.assign_delay_thresh or he[j] > args.assign_h_thresh or ve[j] > args.assign_v_thresh:
                continue
            h0 = int(round(float(ph[i, p]))) % MH
            v0 = int(round(float(pv[i, p]))) % MV
            ds = int(round(float(s) - float(ps[i, p])))
            dh = signed_circular_delta(int(h), h0, MH)
            dv = signed_circular_delta(int(v), v0, MV)
            key = (ds, dh, dv)
            accum[key] += float(val)
            counts[key] += 1
            assigned_power += float(val)

    max_power = max(accum.values()) if accum else 0.0
    offsets = []
    for (ds, dh, dv), power in sorted(accum.items(), key=lambda kv: kv[1], reverse=True):
        if counts[(ds, dh, dv)] < int(args.special_min_offset_count):
            continue
        offsets.append(
            {
                "path_type": 1,
                "offset": [int(ds), int(dh), int(dv)],
                "power": float(power),
                "relative_weight": float(power / max(max_power, 1e-30)),
                "count": int(counts[(ds, dh, dv)]),
                "special_plane_id": int(args.special_plane_id),
            }
        )
    offsets = offsets[: int(args.special_offsets)]
    summary = {
        "special_plane_id": int(args.special_plane_id),
        "special_calibration_top_components": int(args.special_calibration_top_components),
        "special_offsets": offsets,
        "special_offset_count": int(len(offsets)),
        "assigned_top_component_power_frac": float(assigned_power / max(top_power, 1e-30)),
        "top_component_power": float(top_power),
        "assigned_component_power": float(assigned_power),
    }
    return offsets, summary


def gain_for_regime(blocked: int, args) -> tuple[float, str]:
    label = regime_label(int(blocked))
    gains = {
        "clear": float(args.clear_gain),
        "light": float(args.light_gain),
        "medium": float(args.medium_gain),
        "heavy": float(args.heavy_gain),
    }
    return gains[label], label


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
    offsets_by_type: dict[int, list[dict]],
    special_offsets: list[dict],
    args,
) -> tuple[np.ndarray, dict]:
    geom_cfg = make_geom_cfg(args, args.residual_k)
    idx, _raw_d, eff_d = select_geom_neighbors(pos, train_blocked, q_pos, q_blocked, None, geom_cfg)
    weights = idw_weights(eff_d, args.residual_power)

    candidates = []
    q_valid = test_paths["valid"][q_row]
    q_type = test_paths["path_type"][q_row]
    q_plane = test_paths["plane_id"][q_row]
    has_special_path = False
    for p in range(q_valid.shape[0]):
        if not bool(q_valid[p]):
            continue
        typ = int(q_type[p])
        plane = int(q_plane[p])
        is_special = typ == 1 and plane == int(args.special_plane_id) and len(special_offsets) > 0
        if is_special:
            has_special_path = True
            offset_recs = special_offsets
        else:
            offset_recs = offsets_by_type.get(typ, [])
        for offset_rec in offset_recs:
            cand = candidate_coeff(
                p,
                offset_rec,
                test_bins,
                train_bins,
                test_paths,
                train_paths,
                q_row,
                idx,
                weights,
                res_ad,
                args,
            )
            if cand is None:
                continue
            cand["special"] = bool(is_special)
            if is_special:
                cand["score"] *= float(args.special_score_boost)
            candidates.append(cand)

    if not candidates:
        beta_s = beta_subcarriers(args.beta, args.gamma)
        phase = radial_phase(dist_q, beta_s, -1.0)
        return np.zeros((256, 4, S_SC), dtype=np.complex64), {
            "candidate_count": 0,
            "selected_components": 0,
            "selected_score_frac": 0.0,
            "coherence_mean": 0.0,
            "support_mean": 0.0,
            "same_plane_weight_mean": 0.0,
            "fallback_weight_mean": 0.0,
            "reflection_selected_frac": 0.0,
            "special_path": bool(has_special_path),
            "special_selected_frac": 0.0,
            "special_candidate_count": 0,
        }

    candidates.sort(key=lambda c: c["score"], reverse=True)
    selected = candidates[: min(int(args.components), len(candidates))]
    total_score = sum(max(0.0, c["score"]) for c in candidates)
    selected_score = sum(max(0.0, c["score"]) for c in selected)
    pred_ad = np.zeros((1,) + AD_SHAPE, dtype=np.complex64)
    for c in selected:
        pred_ad[0, :, c["h"], c["v"], :, c["s"]] += c["coeff"]

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
        "special_path": bool(has_special_path),
        "special_selected_frac": float(np.mean([bool(c.get("special", False)) for c in selected])) if selected else 0.0,
        "special_candidate_count": int(sum(bool(c.get("special", False)) for c in candidates)),
    }


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
    res_ad = np.load(args.residual_ad_cache, mmap_mode="r")
    res_power = np.load(args.residual_power_cache, mmap_mode="r")
    offsets_by_type, offset_summary = build_offset_table(res_power, paths["train"], args)
    special_offsets, special_summary = build_plane_offset_table(res_power, paths["train"], args)
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
        "special_path_count": 0,
        "special_selected_frac_sum": 0.0,
        "special_candidate_count_sum": 0.0,
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
            offsets_by_type,
            special_offsets,
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
        stats["special_path_count"] += int(meta["special_path"])
        stats["special_selected_frac_sum"] += float(meta["special_selected_frac"])
        stats["special_candidate_count_sum"] += float(meta["special_candidate_count"])

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
        "special_selected_frac",
        "special_candidate_count",
    ]:
        stats[f"{key}_mean"] = stats.pop(f"{key}_sum") / n

    payload = {
        "method": "p51_path_family_expert",
        "version": str(args.version),
        "uses_external_s1": False,
        "formula": "v36 + gain(regime) * inverse_AD(path-relative residual coefficients; P51 uses plane-specific offset expert)",
        "paths": args.paths,
        "v36": args.v36,
        "output": args.output,
        "generic_offset_calibration": offset_summary,
        "special_offset_calibration": special_summary,
        "prediction": {
            "components": int(args.components),
            "offsets_per_type": int(args.offsets_per_type),
            "special_plane_id": int(args.special_plane_id),
            "special_offsets": int(args.special_offsets),
            "special_score_boost": float(args.special_score_boost),
            "calibration_top_components": int(args.calibration_top_components),
            "special_calibration_top_components": int(args.special_calibration_top_components),
            "residual_k": int(args.residual_k),
            "residual_power": float(args.residual_power),
            "same_plane": True,
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
    parser.add_argument("--residual-power-cache", default=str(PATHS.artifacts / "shared_features/train_v36_loo_residual_derot_ad_power.npy"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.version_report_dir(71) / "p51_path_family_expert_v71_generate.json"))
    parser.add_argument("--version", default="v71")
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--h-sign", type=int, default=-1)
    parser.add_argument("--v-sign", type=int, default=1)

    parser.add_argument("--residual-k", type=int, default=32)
    parser.add_argument("--residual-power", type=float, default=2.0)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--offsets-per-type", type=int, default=12)
    parser.add_argument("--calibration-top-components", type=int, default=32)
    parser.add_argument("--min-offset-count", type=int, default=20)
    parser.add_argument("--special-plane-id", type=int, default=51)
    parser.add_argument("--special-offsets", type=int, default=16)
    parser.add_argument("--special-calibration-top-components", type=int, default=32)
    parser.add_argument("--special-min-offset-count", type=int, default=40)
    parser.add_argument("--special-score-boost", type=float, default=1.20)
    parser.add_argument("--assign-delay-sigma", type=float, default=2.0)
    parser.add_argument("--assign-h-sigma", type=float, default=2.5)
    parser.add_argument("--assign-v-sigma", type=float, default=1.5)
    parser.add_argument("--assign-delay-thresh", type=float, default=4.0)
    parser.add_argument("--assign-h-thresh", type=float, default=5.0)
    parser.add_argument("--assign-v-thresh", type=float, default=3.0)
    parser.add_argument("--fallback-same-type", action="store_true")
    parser.add_argument("--fallback-path-bin", action="store_true")
    parser.add_argument("--fallback-weight", type=float, default=0.25)
    parser.add_argument("--fallback-delay-sigma", type=float, default=2.0)
    parser.add_argument("--fallback-h-sigma", type=float, default=2.5)
    parser.add_argument("--fallback-v-sigma", type=float, default=1.5)
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
