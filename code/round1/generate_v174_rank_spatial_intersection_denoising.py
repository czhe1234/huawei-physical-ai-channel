#!/usr/bin/env python3
"""Generate v174: rank-spatial intersection denoising on top of v173.

Hypothesis:
    v173 proved that test-test spatial coherence carries a weak positive
    reliability signal, but the gain was only +0.000011 over v171.  v172
    tests an orthogonal Sionna prior: a clean single path inside one AD cell
    should be closer to low-rank across the 2-pol x 4-UE matrix, while complex
    noise is more full-rank.

    v174 intersects the two signals instead of simply stacking v172 and v173:

    * spatially unsupported weak AD cells are candidate noise;
    * high rank-2 residual inside those cells is treated as noise-like;
    * coherent cells and strong ranked cells are protected;
    * the dominant pol/UE mode is preserved; only the residual mode is shrunk.

This is intentionally more aggressive than v173, but its action is narrower
than v172's broad low-rank denoising.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np

from generate_v87_easy_field_projection import (
    BS_POS,
    MH,
    MV,
    S_SC,
    ad_transform,
    beta_subcarriers,
    inverse_ad,
    summarize,
)
from generate_v159_frontier_consensus_support_soup import compare_outputs, digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import support_metrics_from_cell, top_mask
from generate_v172_rank1_polue_tail_denoising import rank1_projector
from generate_v173_spatial_coherence_tail_denoising import normalize_support
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v174_rank_spatial_intersection_denoising"
OUT_DIR = PATHS.cache_dir(174)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(174)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v174_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v174_rank_spatial_intersection_actions.npz"
REPORT = REPORT_DIR / "v174_rank_spatial_intersection_generate.json"
ZIP_OUT = PATHS.round1_versions / "v174_rank_spatial_intersection_denoising_candidate_pending.zip"

BASE = PATHS.cache_dir(173) / "v173_Round1_Test_Channel.npy"
V171_SUPPORT = PATHS.cache_dir(173) / "v171_collapsed_support.npy"
V173_TARGET = PATHS.cache_dir(173) / "v173_spatial_coherence_target_support.npy"
V172_ACTIONS = PATHS.cache_dir(172) / "v172_rank1_polue_tail_denoising_actions.npz"
V173_ACTIONS = PATHS.cache_dir(173) / "v173_spatial_coherence_tail_denoising_actions.npz"

FLOOR = 1e-12


def lower_power_bounds(regimes: np.ndarray) -> np.ndarray:
    lower = np.ones(len(regimes), dtype=np.float32)
    lower[regimes == "clear"] = 0.9996
    lower[regimes == "light"] = 0.9982
    lower[regimes == "medium"] = 0.9680
    lower[regimes == "heavy"] = 0.9380
    return lower


def build_spatial_gate() -> dict[str, np.ndarray]:
    support = normalize_support(np.load(V171_SUPPORT).astype(np.float32))
    target = normalize_support(np.load(V173_TARGET).astype(np.float32))
    raw_shrink = np.clip(
        (support - target) / np.maximum(support, FLOOR),
        0.0,
        1.0,
    ).astype(np.float32)
    raw_add = np.clip(
        (target - support) / np.maximum(target, FLOOR),
        0.0,
        1.0,
    ).astype(np.float32)
    # Focus on cells v173 actively wanted to attenuate, but keep a soft tail so
    # that uncertain spatial speckles are not missed entirely.
    gate = np.clip((raw_shrink - 0.045) / 0.58, 0.0, 1.0).astype(np.float32)
    gate = np.sqrt(gate).astype(np.float32)
    protect = np.clip(raw_add / 0.35, 0.0, 1.0).astype(np.float32)
    gate *= (1.0 - 0.62 * protect).astype(np.float32)
    return {
        "spatial_gate": gate.astype(np.float32),
        "raw_spatial_shrink": raw_shrink,
        "raw_spatial_add": raw_add,
        "coherent_add_protect": protect,
    }


def build_point_action(regimes: np.ndarray) -> dict[str, np.ndarray]:
    v172 = np.load(V172_ACTIONS)
    v173 = np.load(V173_ACTIONS)
    a172 = v172["action"].astype(np.float32)
    a173 = v173["action"].astype(np.float32)
    # Intersection of two reliability controllers.  The 1.35 multiplier is the
    # aggressive part: if both controllers are high, v174 is allowed to move
    # harder than either conservative selector alone.
    point_action = np.clip(1.35 * np.sqrt(np.maximum(a172 * a173, 0.0)), 0.0, 1.0)
    point_action[regimes == "clear"] = np.minimum(point_action[regimes == "clear"], 0.004)
    point_action[regimes == "light"] = np.minimum(point_action[regimes == "light"], 0.016)

    multibounce_protect = np.minimum(
        v172["multi_protect"].astype(np.float32),
        v173["multibounce_protect"].astype(np.float32),
    )
    point_action = np.clip(point_action * np.clip(multibounce_protect, 0.58, 1.0), 0.0, 1.0)
    return {
        "v172_action": a172,
        "v173_action": a173,
        "point_action": point_action.astype(np.float32),
        "multibounce_protect": multibounce_protect.astype(np.float32),
        "top1_two": v173["top1_two"].astype(np.float32),
        "two_power_frac": v173["two_power_frac"].astype(np.float32),
        "assigned_power_frac": v173["assigned_power_frac"].astype(np.float32),
    }


def denoise_batch(
    ad: np.ndarray,
    point_action: np.ndarray,
    spatial_gate: np.ndarray,
    lower_bound: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    coeff_power = (np.abs(ad) ** 2).astype(np.float32)
    cell = coeff_power.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    metrics = support_metrics_from_cell(cell)
    flat = metrics["flat_support"]

    mask128 = top_mask(flat, 128)
    mask512 = top_mask(flat, 512)
    mask2048 = top_mask(flat, 2048)
    mask4096 = top_mask(flat, 4096)

    base_keep = np.full(flat.shape, 0.010, dtype=np.float32)
    base_keep[mask4096] = 0.080
    base_keep[mask2048] = 0.260
    base_keep[mask512] = 0.720
    base_keep[mask128] = 1.000

    rank_strength = np.ones(flat.shape, dtype=np.float32)
    rank_strength[mask4096] = 0.92
    rank_strength[mask2048] = 0.70
    rank_strength[mask512] = 0.30
    rank_strength[mask128] = 0.0

    a = ad.transpose(0, 2, 3, 5, 1, 4).astype(np.complex64, copy=False)
    rank, residual, residual_ratio = rank1_projector(a)
    residual_gate = np.clip((residual_ratio - 0.028) / 0.205, 0.0, 1.0).astype(np.float32)

    gate_flat = np.clip(spatial_gate.reshape(len(ad), -1), 0.0, 1.0).astype(np.float32)
    effective = np.clip(
        1.55
        * np.sqrt(np.clip(point_action, 0.0, 1.0))[:, None]
        * np.power(gate_flat, 0.62)
        * residual_gate.reshape(len(ad), -1)
        * rank_strength,
        0.0,
        1.0,
    ).astype(np.float32)
    keep_flat = 1.0 - effective * (1.0 - base_keep)
    keep = keep_flat.reshape(len(ad), MH, MV, S_SC).astype(np.float32)

    a1 = rank + np.sqrt(keep)[..., None, None].astype(np.float32) * residual
    ad1 = a1.transpose(0, 4, 1, 2, 5, 3).astype(np.complex64)

    total0 = np.maximum(
        np.sum(np.abs(ad) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64),
        TINY,
    ).astype(np.float32)
    total_direct = np.maximum(
        np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64),
        TINY,
    ).astype(np.float32)
    removed_fraction = np.clip(1.0 - total_direct / total0, 0.0, 1.0).astype(np.float32)
    target_power_factor = np.clip(
        1.0 - 1.05 * removed_fraction,
        lower_bound,
        1.0,
    ).astype(np.float32)
    scale = np.sqrt(target_power_factor * total0 / total_direct).astype(np.float32)
    ad1 *= scale[:, None, None, None, None, None]

    cell1 = (np.abs(ad1) ** 2).astype(np.float32).sum(axis=(1, 4))
    support0 = metrics["support"]
    support1 = cell1 / np.maximum(cell1.sum(axis=(1, 2, 3), keepdims=True), TINY)
    support_l1 = np.sum(np.abs(support1 - support0), axis=(1, 2, 3)).astype(np.float32)

    diag = {
        "top64_fraction": metrics["top64_fraction"],
        "top128_fraction": metrics["top128_fraction"],
        "top512_fraction": metrics["top512_fraction"],
        "tail512_fraction": metrics["tail512_fraction"],
        "entropy": metrics["entropy"],
        "spatial_gate_mean": gate_flat.mean(axis=1).astype(np.float32),
        "spatial_gate_p90": np.quantile(gate_flat, 0.90, axis=1).astype(np.float32),
        "rank2_residual_ratio_mean": residual_ratio.reshape(len(ad), -1).mean(axis=1).astype(np.float32),
        "rank2_residual_ratio_p90": np.quantile(residual_ratio.reshape(len(ad), -1), 0.90, axis=1).astype(np.float32),
        "residual_gate_mean": residual_gate.reshape(len(ad), -1).mean(axis=1).astype(np.float32),
        "effective_mean": effective.mean(axis=1).astype(np.float32),
        "effective_p90": np.quantile(effective, 0.90, axis=1).astype(np.float32),
        "keep_mean": keep.reshape(len(ad), -1).mean(axis=1).astype(np.float32),
        "keep_p10": np.quantile(keep.reshape(len(ad), -1), 0.10, axis=1).astype(np.float32),
        "removed_fraction_direct": removed_fraction,
        "target_power_factor": target_power_factor,
        "actual_power_factor": (
            np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64)
            / np.maximum(total0, TINY)
        ).astype(np.float32),
        "support_l1_to_base": support_l1,
    }
    return ad1, diag


def generate() -> dict[str, object]:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    spatial = build_spatial_gate()
    actions = build_point_action(regimes)
    lower = lower_power_bounds(regimes)

    beta_s = beta_subcarriers(140.3, 0.10)
    distance = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)

    diagnostics: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "top64_fraction",
            "top128_fraction",
            "top512_fraction",
            "tail512_fraction",
            "entropy",
            "spatial_gate_mean",
            "spatial_gate_p90",
            "rank2_residual_ratio_mean",
            "rank2_residual_ratio_p90",
            "residual_gate_mean",
            "effective_mean",
            "effective_p90",
            "keep_mean",
            "keep_p10",
            "removed_fraction_direct",
            "target_power_factor",
            "actual_power_factor",
            "support_l1_to_base",
        )
    }

    for start in range(0, len(source), 20):
        end = min(start + 20, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        derot = h0 * np.exp(
            1j * distance[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        ad0 = ad_transform(derot)
        ad1, diag = denoise_batch(
            ad0,
            actions["point_action"][start:end],
            spatial["spatial_gate"][start:end],
            lower[start:end],
        )
        h1 = inverse_ad(ad1).reshape(end - start, 256, 4, 192)
        h1 = h1 * np.exp(
            -1j * distance[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()
        for key, value in diag.items():
            diagnostics[key].append(value.astype(np.float32))
        print(f"rank-spatial intersection denoised {end}/500", flush=True)

    diag_arrays = {key: np.concatenate(values).astype(np.float32) for key, values in diagnostics.items()}
    spatial_point = {
        "spatial_gate_mean_all": spatial["spatial_gate"].reshape(len(regimes), -1).mean(axis=1).astype(np.float32),
        "raw_spatial_shrink_mean": spatial["raw_spatial_shrink"].reshape(len(regimes), -1).mean(axis=1).astype(np.float32),
        "raw_spatial_add_mean": spatial["raw_spatial_add"].reshape(len(regimes), -1).mean(axis=1).astype(np.float32),
        "coherent_add_protect_mean": spatial["coherent_add_protect"].reshape(len(regimes), -1).mean(axis=1).astype(np.float32),
    }
    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        **actions,
        **spatial_point,
        **diag_arrays,
    )

    comparison_to_v173 = compare_outputs(OUTPUT, BASE, regimes)
    comparison_to_v172 = compare_outputs(OUTPUT, PATHS.cache_dir(172) / "v172_Round1_Test_Channel.npy", regimes)
    comparison_to_v171 = compare_outputs(OUTPUT, PATHS.cache_dir(171) / "v171_Round1_Test_Channel.npy", regimes)
    comparison_to_v170 = compare_outputs(OUTPUT, PATHS.cache_dir(170) / "v170_Round1_Test_Channel.npy", regimes)
    comparison_to_v163 = compare_outputs(OUTPUT, PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy", regimes)
    comparison_to_v140 = compare_outputs(OUTPUT, PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", regimes)
    if comparison_to_v173["shape"] != [500, 256, 4, 192] or comparison_to_v173["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v173['shape']} {comparison_to_v173['dtype']}")
    if not comparison_to_v173["finite"]:
        raise ValueError("v174 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v173 showed spatial coherence is a weak positive reliability signal. "
            "v174 intersects that signal with the v172 low-rank path prior: "
            "only AD cells that are both spatially unsupported and rank-2-rich "
            "are aggressively denoised."
        ),
        "new_information_used": {
            "v173_online": "0.503667 confirms spatial-coherence weak-tail denoising is positive but nearly saturated",
            "v172_rank_prior": "single Sionna path should be low-rank across 2-pol x 4-UE ports",
            "intersection_logic": "noise-like cells should be both spatially incoherent and full-rank; true weak paths should fail at least one gate",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "spatial_gate": "(v171_support - v173_target_support)/v171_support, thresholded at .045 and softened by sqrt",
            "coherent_add_protect": "cells v173 wanted to add are protected from residual shrink",
            "point_action": "1.35 * sqrt(v172_action * v173_action), capped for clear/light and multibounce protected",
            "residual_gate": "clip((rank2_residual_ratio - .028)/.205, 0, 1)",
            "effective": "1.55 * sqrt(point_action) * spatial_gate^.62 * residual_gate * rank_strength",
            "base_keep_residual_power": {
                "top128": 1.00,
                "top512": 0.72,
                "top2048": 0.26,
                "top4096": 0.08,
                "tail": 0.01,
            },
            "target_power_factor": "clip(1 - 1.05*direct_removed_fraction, regime lower bound, 1)",
            "lower_bounds": {"clear": 0.9996, "light": 0.9982, "medium": 0.9680, "heavy": 0.9380},
        },
        "diagnostics": {
            "overall": {
                **{key: summarize(value) for key, value in actions.items() if value.ndim == 1},
                **{key: summarize(value) for key, value in spatial_point.items()},
                **{key: summarize(value) for key, value in diag_arrays.items()},
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "point_action": actions["point_action"],
                    "v172_action": actions["v172_action"],
                    "v173_action": actions["v173_action"],
                    "spatial_gate_mean": diag_arrays["spatial_gate_mean"],
                    "rank2_residual_ratio_mean": diag_arrays["rank2_residual_ratio_mean"],
                    "effective_mean": diag_arrays["effective_mean"],
                    "removed_fraction_direct": diag_arrays["removed_fraction_direct"],
                    "target_power_factor": diag_arrays["target_power_factor"],
                    "actual_power_factor": diag_arrays["actual_power_factor"],
                    "support_l1_to_base": diag_arrays["support_l1_to_base"],
                },
            ),
        },
        "comparison_to_v173": comparison_to_v173,
        "comparison_to_v172": comparison_to_v172,
        "comparison_to_v171": comparison_to_v171,
        "comparison_to_v170": comparison_to_v170,
        "comparison_to_v163": comparison_to_v163,
        "comparison_to_v140": comparison_to_v140,
        "preserved": {
            "base_complex": "v173",
            "dominant_polue_mode": "rank-1 dominant mode preserved in each AD cell",
            "coherent_spatial_cells": "v173 add/coherent cells are protected",
            "strong_cells": "top128 fully protected; top512 mostly protected",
            "randomness": "none",
        },
        "changed": {
            "unsupported_full_rank_tail": "rank-2 residual mode shrink in spatially incoherent weak cells",
            "total_power": "bounded shrink from residual-mode removed mass",
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source, name in (
            (__file__, "generate_v174_rank_spatial_intersection_denoising.py"),
            (PATHS.code / "generate_v173_spatial_coherence_tail_denoising.py", "generate_v173_spatial_coherence_tail_denoising.py"),
            (PATHS.code / "generate_v172_rank1_polue_tail_denoising.py", "generate_v172_rank1_polue_tail_denoising.py"),
            (PATHS.code / "generate_v171_full_complex_ad_tail_denoising.py", "generate_v171_full_complex_ad_tail_denoising.py"),
            (PATHS.code / "generate_v170_adaptive_floor_subtraction_denoising.py", "generate_v170_adaptive_floor_subtraction_denoising.py"),
            (PATHS.code / "generate_v169_power_scaled_tail_denoising.py", "generate_v169_power_scaled_tail_denoising.py"),
            (PATHS.code / "analyze_v168_channel_awgn_power_audit.py", "analyze_v168_channel_awgn_power_audit.py"),
            (PATHS.code / "analyze_v166_path_family_predictability.py", "analyze_v166_path_family_predictability.py"),
            (PATHS.code / "generate_v159_frontier_consensus_support_soup.py", "generate_v159_frontier_consensus_support_soup.py"),
            (PATHS.code / "generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
            (PATHS.code / "powerfield_common.py", "powerfield_common.py"),
            (PATHS.code_root / "project_paths.py", "project_paths.py"),
        ):
            archive.write(source, name)

    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise ValueError(f"ZIP CRC failure: {bad_member}")
    output = np.load(OUTPUT, mmap_mode="r")
    integrity = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(np.isfinite(output.real).all() and np.isfinite(output.imag).all()),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    report["integrity"] = integrity
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    report = generate()
    report = write_zip(report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
