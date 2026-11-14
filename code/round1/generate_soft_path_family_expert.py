#!/usr/bin/env python3
"""Generate a soft multi-family path residual expert candidate.

v68 uses one generic reflection offset table for all reflection planes.  v71
replaced that generic table with a P51-specific table when the query path used
plane 51, but online feedback showed that hard replacement was too narrow.

This version keeps the v68 generic candidates and adds plane-specific expert
candidates for several high-support reflection planes.  The experts compete
softly with the generic candidates instead of replacing them.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from build_map_features import BS_POS
from evaluate_local_ad_paths import S_SC, inverse_ad
from evaluate_s1style_geom_blend import geometry_cache, select_geom_neighbors
from generate_p51_path_family_expert import build_plane_offset_table
from generate_path_relative_residual_prior import (
    build_offset_table,
    candidate_coeff,
    load_paths,
    path_bins,
)
from generate_train_residual_ad_prior import (
    AD_SHAPE,
    beta_subcarriers,
    idw_weights,
    make_geom_cfg,
    radial_phase,
    regime_label,
)


REGIMES = ("clear", "light", "medium", "heavy")


def parse_plane_ids(text: str) -> list[int]:
    ids: list[int] = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids


def gain_for_regime(blocked: int, args) -> tuple[float, str]:
    label = regime_label(int(blocked))
    gains = {
        "clear": float(args.clear_gain),
        "light": float(args.light_gain),
        "medium": float(args.medium_gain),
        "heavy": float(args.heavy_gain),
    }
    return gains[label], label


def build_expert_tables(res_power, train_paths: dict[str, np.ndarray], args) -> tuple[dict[int, list[dict]], dict]:
    """Build plane-specific offset experts for selected reflection planes."""

    experts: dict[int, list[dict]] = {}
    summary: dict[str, dict] = {}
    for plane_id in parse_plane_ids(args.expert_planes):
        local_args = SimpleNamespace(**vars(args))
        local_args.special_plane_id = int(plane_id)
        offsets, info = build_plane_offset_table(res_power, train_paths, local_args)
        if offsets:
            experts[int(plane_id)] = offsets
        summary[str(int(plane_id))] = info
    return experts, summary


def expert_quality(expert_info: dict, args) -> float:
    assigned = float(expert_info.get("assigned_top_component_power_frac", 0.0))
    count = float(expert_info.get("special_offset_count", 0))
    assigned_gate = np.clip(assigned / max(float(args.expert_assigned_ref), 1e-12), 0.0, 1.5)
    count_gate = np.clip(count / max(float(args.expert_count_ref), 1e-12), 0.0, 1.0)
    quality = (assigned_gate ** float(args.expert_assigned_power)) * (count_gate ** float(args.expert_count_power))
    return float(np.clip(quality, 0.0, float(args.expert_quality_cap)))


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
    expert_tables: dict[int, list[dict]],
    expert_summary: dict[str, dict],
    args,
) -> tuple[np.ndarray, dict]:
    geom_cfg = make_geom_cfg(args, args.residual_k)
    idx, _raw_d, eff_d = select_geom_neighbors(pos, train_blocked, q_pos, q_blocked, None, geom_cfg)
    weights = idw_weights(eff_d, args.residual_power)

    candidates = []
    q_valid = test_paths["valid"][q_row]
    q_type = test_paths["path_type"][q_row]
    q_plane = test_paths["plane_id"][q_row]
    expert_path_count = 0
    expert_candidate_count = 0

    for p in range(q_valid.shape[0]):
        if not bool(q_valid[p]):
            continue
        typ = int(q_type[p])
        plane = int(q_plane[p])

        # Always keep the generic v68 path-relative candidates.
        for offset_rec in offsets_by_type.get(typ, []):
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
            if cand is not None:
                cand["expert"] = False
                cand["expert_plane"] = -1
                candidates.append(cand)

        # Add plane-specific candidates as soft competitors.
        if typ == 1 and plane in expert_tables:
            expert_path_count += 1
            quality = expert_quality(expert_summary.get(str(plane), {}), args)
            path_score = float(test_paths["score"][q_row, p])
            path_gate = float(1.0 / (1.0 + np.exp(-float(args.expert_path_gate_temp) * (path_score - float(args.expert_path_gate_center)))))
            expert_boost = float(args.expert_score_boost) * quality * path_gate
            if expert_boost <= 0.0:
                continue
            for offset_rec in expert_tables[plane]:
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
                cand["score"] *= expert_boost
                cand["expert"] = True
                cand["expert_plane"] = int(plane)
                cand["expert_boost"] = float(expert_boost)
                candidates.append(cand)
                expert_candidate_count += 1

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
            "expert_path_count": int(expert_path_count),
            "expert_candidate_count": int(expert_candidate_count),
            "expert_selected_frac": 0.0,
            "expert_boost_mean": 0.0,
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
    expert_selected = [bool(c.get("expert", False)) for c in selected]
    expert_boosts = [float(c.get("expert_boost", 0.0)) for c in selected if bool(c.get("expert", False))]
    return residual, {
        "candidate_count": int(len(candidates)),
        "selected_components": int(len(selected)),
        "selected_score_frac": float(selected_score / max(total_score, 1e-30)),
        "coherence_mean": float(np.mean([c["coherence"] for c in selected])) if selected else 0.0,
        "support_mean": float(np.mean([c["support"] for c in selected])) if selected else 0.0,
        "same_plane_weight_mean": float(np.mean([c["same_plane_weight"] for c in selected])) if selected else 0.0,
        "fallback_weight_mean": float(np.mean([c["fallback_weight"] for c in selected])) if selected else 0.0,
        "reflection_selected_frac": float(np.mean([c["path_type"] == 1 for c in selected])) if selected else 0.0,
        "expert_path_count": int(expert_path_count),
        "expert_candidate_count": int(expert_candidate_count),
        "expert_selected_frac": float(np.mean(expert_selected)) if selected else 0.0,
        "expert_boost_mean": float(np.mean(expert_boosts)) if expert_boosts else 0.0,
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
    expert_tables, expert_summary = build_expert_tables(res_power, paths["train"], args)
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
        "expert_path_count_sum": 0.0,
        "expert_candidate_count_sum": 0.0,
        "expert_selected_frac_sum": 0.0,
        "expert_boost_sum": 0.0,
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
            expert_tables,
            expert_summary,
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
        for key in [
            "candidate_count",
            "selected_components",
            "selected_score_frac",
            "coherence_mean",
            "support_mean",
            "same_plane_weight_mean",
            "fallback_weight_mean",
            "reflection_selected_frac",
            "expert_path_count",
            "expert_candidate_count",
            "expert_selected_frac",
            "expert_boost_mean",
        ]:
            stat_key = key.replace("_mean", "")
            stats[f"{stat_key}_sum"] += float(meta[key])

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
        "expert_path_count",
        "expert_candidate_count",
        "expert_selected_frac",
        "expert_boost",
    ]:
        stats[f"{key}_mean"] = stats.pop(f"{key}_sum") / n

    payload = {
        "method": "soft_path_family_expert",
        "version": str(args.version),
        "uses_external_s1": False,
        "formula": "v36 + gain(regime) * inverse_AD(generic path-relative residual + soft plane-specific experts)",
        "paths": args.paths,
        "v36": args.v36,
        "output": args.output,
        "generic_offset_calibration": offset_summary,
        "expert_offset_calibration": expert_summary,
        "prediction": {
            "components": int(args.components),
            "offsets_per_type": int(args.offsets_per_type),
            "expert_planes": parse_plane_ids(args.expert_planes),
            "expert_offsets": int(args.special_offsets),
            "expert_score_boost": float(args.expert_score_boost),
            "expert_assigned_ref": float(args.expert_assigned_ref),
            "expert_count_ref": float(args.expert_count_ref),
            "expert_quality_cap": float(args.expert_quality_cap),
            "calibration_top_components": int(args.calibration_top_components),
            "special_calibration_top_components": int(args.special_calibration_top_components),
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
    parser.add_argument("--residual-power-cache", default=str(PATHS.artifacts / "shared_features/train_v36_loo_residual_derot_ad_power.npy"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.version_report_dir(72) / "soft_path_family_expert_v72_generate.json"))
    parser.add_argument("--version", default="v72")
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
    parser.add_argument("--expert-planes", default="51,0,2,5,58,90,25,92")
    parser.add_argument("--special-offsets", type=int, default=12)
    parser.add_argument("--special-calibration-top-components", type=int, default=32)
    parser.add_argument("--special-min-offset-count", type=int, default=50)
    parser.add_argument("--expert-score-boost", type=float, default=0.90)
    parser.add_argument("--expert-assigned-ref", type=float, default=0.55)
    parser.add_argument("--expert-count-ref", type=float, default=10.0)
    parser.add_argument("--expert-assigned-power", type=float, default=0.75)
    parser.add_argument("--expert-count-power", type=float, default=0.25)
    parser.add_argument("--expert-quality-cap", type=float, default=1.15)
    parser.add_argument("--expert-path-gate-center", type=float, default=0.0)
    parser.add_argument("--expert-path-gate-temp", type=float, default=1.0)
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
