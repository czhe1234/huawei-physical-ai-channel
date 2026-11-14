#!/usr/bin/env python3
"""Generate v172: rank-1 pol/UE tail denoising on top of v171.

Hypothesis:
    v171 confirmed that weak AD cells' full complex coefficients can be
    denoised.  It still treats the 2-pol x 4-UE coefficients inside each
    angle-delay cell mostly as independent powers.  A single physical Sionna
    path landing in one AD cell should be closer to a low-rank complex matrix
    across polarization and UE ports, while complex Gaussian-like noise is more
    full-rank.  v172 therefore shrinks only the non-dominant polar/UE mode of
    weak AD cells, preserving the dominant path mode and strong cells.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np

from generate_v87_easy_field_projection import (
    BS_POS,
    MH,
    MP,
    MV,
    N_ANT,
    S_SC,
    ad_transform,
    beta_subcarriers,
    inverse_ad,
    summarize,
)
from generate_v159_frontier_consensus_support_soup import compare_outputs, digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import support_metrics_from_cell, top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v172_rank1_polue_tail_denoising"
OUT_DIR = PATHS.cache_dir(172)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(172)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v172_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v172_rank1_polue_tail_denoising_actions.npz"
REPORT = REPORT_DIR / "v172_rank1_polue_tail_denoising_generate.json"
ZIP_OUT = PATHS.round1_versions / "v172_rank1_polue_tail_denoising_candidate_pending.zip"

BASE = PATHS.cache_dir(171) / "v171_Round1_Test_Channel.npy"
V171_ACTIONS = PATHS.cache_dir(171) / "v171_full_complex_ad_tail_denoising_actions.npz"
V170_ACTIONS = PATHS.cache_dir(170) / "v170_adaptive_floor_subtraction_actions.npz"

CELL_COUNT = MH * MV * S_SC
FLOOR = 1e-12


def lower_power_bounds(regimes: np.ndarray) -> np.ndarray:
    lower = np.ones(len(regimes), dtype=np.float32)
    lower[regimes == "clear"] = 0.999
    lower[regimes == "light"] = 0.997
    lower[regimes == "medium"] = 0.955
    lower[regimes == "heavy"] = 0.925
    return lower


def build_action(regimes: np.ndarray) -> dict[str, np.ndarray]:
    v171 = np.load(V171_ACTIONS)
    v170 = np.load(V170_ACTIONS)
    base = v171["action"].astype(np.float32)

    scale = np.ones(len(base), dtype=np.float32)
    scale[regimes == "clear"] = 0.10
    scale[regimes == "light"] = 0.22
    scale[regimes == "medium"] = 1.65
    scale[regimes == "heavy"] = 1.95

    tail = v171["tail512_fraction"].astype(np.float32)
    entropy = v171["entropy"].astype(np.float32)
    tail_gate = np.clip((tail - 0.018) / 0.075, 0.0, 1.0).astype(np.float32)
    entropy_gate = np.clip((entropy - 0.36) / 0.22, 0.0, 1.0).astype(np.float32)

    top1_two = v170["top1_two"].astype(np.float32)
    two_power = v170["two_power_frac"].astype(np.float32)
    assigned = v170["assigned_power_frac"].astype(np.float32)
    multi_protect = (
        1.20
        - 0.32 * np.clip((top1_two - 0.68) / 0.26, 0.0, 1.0)
        - 0.20 * np.clip((two_power - 0.45) / 0.30, 0.0, 1.0)
        - 0.10 * np.clip((assigned - 0.78) / 0.14, 0.0, 1.0)
    ).astype(np.float32)
    multi_protect = np.clip(multi_protect, 0.55, 1.20)

    unexplained_gate = np.clip((0.86 - assigned) / 0.55, 0.0, 1.0).astype(np.float32)
    noise_gate = np.clip(
        0.58 + 0.22 * tail_gate + 0.20 * entropy_gate + 0.18 * unexplained_gate,
        0.40,
        1.25,
    ).astype(np.float32)

    action = np.clip(base * scale * multi_protect * noise_gate, 0.0, 1.0).astype(np.float32)
    action[regimes == "clear"] = np.minimum(action[regimes == "clear"], 0.006)
    action[regimes == "light"] = np.minimum(action[regimes == "light"], 0.025)

    return {
        "v171_action": base,
        "action_scale": scale,
        "tail_gate": tail_gate,
        "entropy_gate": entropy_gate,
        "multi_protect": multi_protect,
        "unexplained_gate": unexplained_gate,
        "noise_gate": noise_gate,
        "action": action,
        "top1_two": top1_two,
        "two_power_frac": two_power,
        "assigned_power_frac": assigned,
    }


def rank1_projector(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dominant-mode part, residual, and residual power ratio.

    `a` has shape (B, H, V, S, 2, 4).  The projector is computed from the 2x2
    receive-polarization Gram matrix for each AD cell.  For nearly isotropic
    cells the dominant direction is not well-defined; those cells fall back to
    a neutral 0.5/0.5 split, which behaves like residual-only shrink without a
    hard polarization bias.
    """
    a0 = a[..., 0, :]
    a1 = a[..., 1, :]
    g00 = np.sum(np.abs(a0) ** 2, axis=-1).astype(np.float32)
    g11 = np.sum(np.abs(a1) ** 2, axis=-1).astype(np.float32)
    g01 = np.sum(a0 * np.conj(a1), axis=-1).astype(np.complex64)

    trace = np.maximum(g00 + g11, TINY).astype(np.float32)
    delta = np.sqrt(np.maximum((g00 - g11) ** 2 + 4.0 * (np.abs(g01) ** 2), 0.0)).astype(np.float32)
    lam2 = np.maximum(0.5 * (trace - delta), 0.0).astype(np.float32)
    residual_ratio = np.clip(lam2 / trace, 0.0, 0.5).astype(np.float32)

    denom = np.maximum(delta, 1e-30).astype(np.float32)
    p00 = ((g00 - lam2) / denom).astype(np.float32)
    p11 = ((g11 - lam2) / denom).astype(np.float32)
    p01 = (g01 / denom).astype(np.complex64)
    near_iso = (delta / trace) < 1e-5
    if np.any(near_iso):
        p00 = np.where(near_iso, 0.5, p00).astype(np.float32)
        p11 = np.where(near_iso, 0.5, p11).astype(np.float32)
        p01 = np.where(near_iso, 0.0 + 0.0j, p01).astype(np.complex64)
    p10 = np.conj(p01)

    rank0 = p00[..., None] * a0 + p01[..., None] * a1
    rank1 = p10[..., None] * a0 + p11[..., None] * a1
    rank = np.stack([rank0, rank1], axis=-2).astype(np.complex64)
    residual = (a - rank).astype(np.complex64)
    return rank, residual, residual_ratio


def denoise_batch(
    ad: np.ndarray,
    action: np.ndarray,
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

    base_keep = np.full(flat.shape, 0.006, dtype=np.float32)
    base_keep[mask4096] = 0.055
    base_keep[mask2048] = 0.180
    base_keep[mask512] = 0.600
    base_keep[mask128] = 1.000

    a = ad.transpose(0, 2, 3, 5, 1, 4).astype(np.complex64, copy=False)
    rank, residual, residual_ratio = rank1_projector(a)
    residual_gate = np.clip((residual_ratio - 0.035) / 0.200, 0.0, 1.0).astype(np.float32)
    strength = np.sqrt(np.clip(action, 0.0, 1.0)).astype(np.float32)
    effective = np.clip(strength[:, None] * residual_gate.reshape(len(ad), -1), 0.0, 1.0)
    keep_flat = 1.0 - effective * (1.0 - base_keep)
    keep = keep_flat.reshape(len(ad), MH, MV, S_SC).astype(np.float32)

    a1 = rank + np.sqrt(keep)[..., None, None].astype(np.float32) * residual
    ad1 = a1.transpose(0, 4, 1, 2, 5, 3).astype(np.complex64)

    total0 = np.maximum(np.sum(np.abs(ad) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64), TINY).astype(np.float32)
    total_direct = np.maximum(np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64), TINY).astype(np.float32)
    removed_fraction = np.clip(1.0 - total_direct / total0, 0.0, 1.0).astype(np.float32)
    target_power_factor = np.clip(1.0 - 1.22 * removed_fraction, lower_bound, 1.0).astype(np.float32)
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
        "rank2_residual_ratio_mean": residual_ratio.reshape(len(ad), -1).mean(axis=1).astype(np.float32),
        "rank2_residual_ratio_p90": np.quantile(residual_ratio.reshape(len(ad), -1), 0.90, axis=1).astype(np.float32),
        "residual_gate_mean": residual_gate.reshape(len(ad), -1).mean(axis=1).astype(np.float32),
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
    blocked_test = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "test_blocked"
    ].astype(np.float32)
    regimes = regime_of(blocked_test)
    actions = build_action(regimes)
    lower = lower_power_bounds(regimes)

    beta_s = beta_subcarriers(140.3, 0.10)
    distance = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )

    diagnostics: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "top64_fraction",
            "top128_fraction",
            "top512_fraction",
            "tail512_fraction",
            "entropy",
            "rank2_residual_ratio_mean",
            "rank2_residual_ratio_p90",
            "residual_gate_mean",
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
        ad1, diag = denoise_batch(ad0, actions["action"][start:end], lower[start:end])
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
        print(f"rank1 pol/UE denoised {end}/500", flush=True)

    diag_arrays = {key: np.concatenate(values).astype(np.float32) for key, values in diagnostics.items()}
    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        **actions,
        **diag_arrays,
    )

    comparison_to_v171 = compare_outputs(OUTPUT, BASE, regimes)
    comparison_to_v170 = compare_outputs(
        OUTPUT, PATHS.cache_dir(170) / "v170_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v169 = compare_outputs(
        OUTPUT, PATHS.cache_dir(169) / "v169_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v163 = compare_outputs(
        OUTPUT, PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v140 = compare_outputs(
        OUTPUT, PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", regimes
    )
    if comparison_to_v171["shape"] != [500, 256, 4, 192] or comparison_to_v171["dtype"] != "complex64":
        raise ValueError(
            f"invalid output layout: {comparison_to_v171['shape']} {comparison_to_v171['dtype']}"
        )
    if not comparison_to_v171["finite"]:
        raise ValueError("v172 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v171 showed weak full-complex AD coefficients can be denoised. "
            "v172 tests a deeper Sionna prior: a real path inside one AD cell "
            "should be closer to rank-1 across the 2-pol x 4-UE matrix, while "
            "noise is more full-rank.  Therefore only weak cells' non-dominant "
            "polar/UE mode is shrunk, with strong cells and likely multipath "
            "points protected."
        ),
        "new_information_used": {
            "v171_online": "0.503656 confirms full-complex AD weak-tail denoising improves v170",
            "sionna_path_prior": "single path component should be low-rank across pol/UE ports",
            "v166_path_family": "top1_two/two_power/assigned_power protect likely real multipath from over-denoising",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "action": "v171 action scaled by regime and path/noise gates",
            "regime_scale": {"clear": 0.10, "light": 0.22, "medium": 1.65, "heavy": 1.95},
            "action_caps": {"clear": 0.006, "light": 0.025},
            "rank_prior": "2x4 AD-cell matrix -> dominant polar mode + residual; shrink residual only",
            "base_keep_residual_power": {
                "top128": 1.00,
                "top512": 0.60,
                "top2048": 0.18,
                "top4096": 0.055,
                "tail": 0.006,
            },
            "residual_gate": "clip((rank2_residual_ratio - .035)/.200, 0, 1)",
            "target_power_factor": "clip(1 - 1.22*direct_removed_fraction, regime lower bound, 1)",
            "lower_bounds": {"clear": 0.999, "light": 0.997, "medium": 0.955, "heavy": 0.925},
        },
        "diagnostics": {
            "overall": {
                **{key: summarize(value) for key, value in actions.items() if value.ndim == 1},
                **{key: summarize(value) for key, value in diag_arrays.items()},
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "action": actions["action"],
                    "v171_action": actions["v171_action"],
                    "multi_protect": actions["multi_protect"],
                    "noise_gate": actions["noise_gate"],
                    "rank2_residual_ratio_mean": diag_arrays["rank2_residual_ratio_mean"],
                    "residual_gate_mean": diag_arrays["residual_gate_mean"],
                    "keep_mean": diag_arrays["keep_mean"],
                    "removed_fraction_direct": diag_arrays["removed_fraction_direct"],
                    "target_power_factor": diag_arrays["target_power_factor"],
                    "actual_power_factor": diag_arrays["actual_power_factor"],
                    "support_l1_to_base": diag_arrays["support_l1_to_base"],
                },
            ),
        },
        "comparison_to_v171": comparison_to_v171,
        "comparison_to_v170": comparison_to_v170,
        "comparison_to_v169": comparison_to_v169,
        "comparison_to_v163": comparison_to_v163,
        "comparison_to_v140": comparison_to_v140,
        "preserved": {
            "base_complex": "v171",
            "strong_cells": "top128 residual power fully preserved; top512 strongly protected",
            "dominant_polue_mode": "rank-1 dominant mode is preserved in every AD cell",
            "randomness": "none",
        },
        "changed": {
            "weak_cell_internal_structure": "non-dominant pol/UE mode shrink",
            "weak_tail_phase_and_pol_ue": "full-rank noise-like component attenuated",
            "total_power": "bounded shrink from direct removed residual-mode power",
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
            (__file__, "generate_v172_rank1_polue_tail_denoising.py"),
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
