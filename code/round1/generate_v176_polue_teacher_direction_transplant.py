#!/usr/bin/env python3
"""Generate v176: transplant v172 pol/UE direction into v173 support.

Hypothesis:
    v169-v173 mostly improved AD support, weak-tail power and reliability.
    They deliberately preserved most cell-internal complex direction across
    the 2-pol x 4-UE ports.  v172 is the first candidate that changes that
    internal direction through a low-rank Sionna path prior, but it also changes
    power and support.

    v176 isolates the missing information: keep v173 AD cell support and total
    power, but rotate weak medium/heavy cells' 2x4 complex direction toward
    v172's low-rank teacher direction.  If score is capped by phase/pol/UE
    allocation rather than support, this should move more than v175 without
    disturbing v173's proven PAS/PDP-like skeleton.
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
from generate_v174_rank_spatial_intersection_denoising import build_point_action, build_spatial_gate
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v176_polue_teacher_direction_transplant"
OUT_DIR = PATHS.cache_dir(176)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(176)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v176_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v176_polue_teacher_direction_transplant_actions.npz"
REPORT = REPORT_DIR / "v176_polue_teacher_direction_transplant_generate.json"
ZIP_OUT = PATHS.round1_versions / "v176_polue_teacher_direction_transplant_candidate_pending.zip"

BASE = PATHS.cache_dir(173) / "v173_Round1_Test_Channel.npy"
TEACHER = PATHS.cache_dir(172) / "v172_Round1_Test_Channel.npy"
EPS = 1e-18


def transplant_batch(
    ad_base: np.ndarray,
    ad_teacher: np.ndarray,
    point_action: np.ndarray,
    spatial_gate: np.ndarray,
    add_protect: np.ndarray,
    regimes: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cell = (np.abs(ad_base) ** 2).astype(np.float32).sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    metrics = support_metrics_from_cell(cell)
    flat = metrics["flat_support"]

    mask128 = top_mask(flat, 128)
    mask512 = top_mask(flat, 512)
    mask2048 = top_mask(flat, 2048)
    mask4096 = top_mask(flat, 4096)

    rank_strength = np.full(flat.shape, 0.86, dtype=np.float32)
    rank_strength[mask4096] = 0.72
    rank_strength[mask2048] = 0.48
    rank_strength[mask512] = 0.16
    rank_strength[mask128] = 0.0

    gate_flat = np.clip(spatial_gate.reshape(len(ad_base), -1), 0.0, 1.0).astype(np.float32)
    protect_flat = np.clip(add_protect.reshape(len(ad_base), -1), 0.0, 1.0).astype(np.float32)
    spatial_conf = (0.30 + 0.70 * np.power(gate_flat, 0.55)).astype(np.float32)
    coherent_protect = np.clip(1.0 - 0.58 * protect_flat, 0.32, 1.0).astype(np.float32)

    regime_gain = np.ones(len(ad_base), dtype=np.float32)
    regime_gain[regimes == "clear"] = 0.0
    regime_gain[regimes == "light"] = 0.04
    regime_gain[regimes == "medium"] = 0.78
    regime_gain[regimes == "heavy"] = 1.05

    w = np.clip(
        1.18
        * np.sqrt(np.clip(point_action, 0.0, 1.0))[:, None]
        * regime_gain[:, None]
        * spatial_conf
        * coherent_protect
        * rank_strength,
        0.0,
        0.88,
    ).astype(np.float32)

    base_vec = ad_base.transpose(0, 2, 3, 5, 1, 4).astype(np.complex64, copy=False)
    teacher_vec = ad_teacher.transpose(0, 2, 3, 5, 1, 4).astype(np.complex64, copy=False)
    base_norm = np.sqrt(np.sum(np.abs(base_vec) ** 2, axis=(-2, -1), keepdims=True) + EPS).astype(np.float32)
    teacher_norm = np.sqrt(np.sum(np.abs(teacher_vec) ** 2, axis=(-2, -1), keepdims=True) + EPS).astype(np.float32)
    d0 = base_vec / base_norm
    d1 = teacher_vec / teacher_norm

    inner = np.sum(np.conj(d1) * d0, axis=(-2, -1), keepdims=True)
    phase = inner / np.maximum(np.abs(inner), EPS)
    d1_aligned = d1 * phase
    teacher_cos = np.abs(inner).reshape(len(ad_base), -1).astype(np.float32)
    teacher_conf = np.clip((teacher_cos - 0.10) / 0.72, 0.0, 1.0).astype(np.float32)
    # Keep this aggressive: low teacher agreement reduces but does not kill the
    # action, because the whole point is to test whether v172's direction has
    # missing information not present in v173.
    w *= (0.42 + 0.58 * np.sqrt(teacher_conf)).astype(np.float32)

    w_cell = w.reshape(len(ad_base), MH, MV, S_SC).astype(np.float32)
    mix = (1.0 - w_cell[..., None, None]) * d0 + w_cell[..., None, None] * d1_aligned
    mix_norm = np.sqrt(np.sum(np.abs(mix) ** 2, axis=(-2, -1), keepdims=True) + EPS).astype(np.float32)
    mix_dir = mix / mix_norm
    out_vec = mix_dir * base_norm
    ad_out = out_vec.transpose(0, 4, 1, 2, 5, 3).astype(np.complex64)

    total0 = np.maximum(
        np.sum(np.abs(ad_base) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64),
        TINY,
    ).astype(np.float32)
    total1 = np.maximum(
        np.sum(np.abs(ad_out) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64),
        TINY,
    ).astype(np.float32)
    ad_out *= np.sqrt(total0 / total1)[:, None, None, None, None, None].astype(np.float32)

    diag = {
        "top64_fraction": metrics["top64_fraction"],
        "top128_fraction": metrics["top128_fraction"],
        "top512_fraction": metrics["top512_fraction"],
        "tail512_fraction": metrics["tail512_fraction"],
        "entropy": metrics["entropy"],
        "spatial_gate_mean": gate_flat.mean(axis=1).astype(np.float32),
        "add_protect_mean": protect_flat.mean(axis=1).astype(np.float32),
        "teacher_cos_mean": teacher_cos.mean(axis=1).astype(np.float32),
        "teacher_cos_p10": np.quantile(teacher_cos, 0.10, axis=1).astype(np.float32),
        "teacher_conf_mean": teacher_conf.mean(axis=1).astype(np.float32),
        "weight_mean": w.mean(axis=1).astype(np.float32),
        "weight_p90": np.quantile(w, 0.90, axis=1).astype(np.float32),
        "actual_power_factor": (
            np.sum(np.abs(ad_out) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64)
            / np.maximum(total0, TINY)
        ).astype(np.float32),
    }
    return ad_out, diag


def generate() -> dict[str, object]:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    spatial = build_spatial_gate()
    actions = build_point_action(regimes)

    beta_s = beta_subcarriers(140.3, 0.10)
    distance = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    source = np.load(BASE, mmap_mode="r")
    teacher = np.load(TEACHER, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)

    diagnostics: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "top64_fraction",
            "top128_fraction",
            "top512_fraction",
            "tail512_fraction",
            "entropy",
            "spatial_gate_mean",
            "add_protect_mean",
            "teacher_cos_mean",
            "teacher_cos_p10",
            "teacher_conf_mean",
            "weight_mean",
            "weight_p90",
            "actual_power_factor",
        )
    }

    for start in range(0, len(source), 20):
        end = min(start + 20, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        ht = np.asarray(teacher[start:end], dtype=np.complex64)
        rot = np.exp(1j * distance[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad0 = ad_transform(h0 * rot)
        adt = ad_transform(ht * rot)
        ad1, diag = transplant_batch(
            ad0,
            adt,
            actions["point_action"][start:end],
            spatial["spatial_gate"][start:end],
            spatial["coherent_add_protect"][start:end],
            regimes[start:end],
        )
        h1 = inverse_ad(ad1).reshape(end - start, 256, 4, 192)
        h1 = h1 * np.exp(-1j * distance[start:end, None, None, None] * beta_s).astype(np.complex64)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()
        for key, value in diag.items():
            diagnostics[key].append(value.astype(np.float32))
        print(f"polue teacher direction transplanted {end}/500", flush=True)

    diag_arrays = {key: np.concatenate(values).astype(np.float32) for key, values in diagnostics.items()}
    spatial_point = {
        "spatial_gate_mean_all": spatial["spatial_gate"].reshape(len(regimes), -1).mean(axis=1).astype(np.float32),
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
    comparison_to_v175 = compare_outputs(OUTPUT, PATHS.cache_dir(175) / "v175_Round1_Test_Channel.npy", regimes)
    comparison_to_v172 = compare_outputs(OUTPUT, TEACHER, regimes)
    comparison_to_v171 = compare_outputs(OUTPUT, PATHS.cache_dir(171) / "v171_Round1_Test_Channel.npy", regimes)
    comparison_to_v163 = compare_outputs(OUTPUT, PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy", regimes)
    if comparison_to_v173["shape"] != [500, 256, 4, 192] or comparison_to_v173["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v173['shape']} {comparison_to_v173['dtype']}")
    if not comparison_to_v173["finite"]:
        raise ValueError("v176 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "Keep v173 support and total power, but transplant weak-cell 2-pol x "
            "4-UE complex direction toward v172's low-rank teacher. This isolates "
            "cell-internal phase/pol/UE allocation as a possible missing score source."
        ),
        "new_information_used": {
            "v173_online": "best known support/power skeleton",
            "v172_teacher": "low-rank pol/UE direction proposal",
            "isolation": "preserve AD cell power and point total power to avoid conflating direction with support/power",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "teacher": PATHS.relative(TEACHER),
            "cell_power": "preserve v173 cell norm; final point total power renormalized to v173",
            "teacher_alignment": "per-cell complex inner product phase aligns teacher direction to v173 before blending",
            "weight": "medium/heavy weak cells only; top128 locked, top512 small, coherent-add cells protected",
            "max_weight": 0.88,
            "randomness": "none",
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
                    "spatial_gate_mean": diag_arrays["spatial_gate_mean"],
                    "add_protect_mean": diag_arrays["add_protect_mean"],
                    "teacher_cos_mean": diag_arrays["teacher_cos_mean"],
                    "teacher_conf_mean": diag_arrays["teacher_conf_mean"],
                    "weight_mean": diag_arrays["weight_mean"],
                    "weight_p90": diag_arrays["weight_p90"],
                    "actual_power_factor": diag_arrays["actual_power_factor"],
                },
            ),
        },
        "comparison_to_v173": comparison_to_v173,
        "comparison_to_v175": comparison_to_v175,
        "comparison_to_v172": comparison_to_v172,
        "comparison_to_v171": comparison_to_v171,
        "comparison_to_v163": comparison_to_v163,
        "preserved": {
            "base_support": "v173 AD cell power/support",
            "base_total_power": "v173 point total power",
            "strong_cells": "top128 fully locked; top512 low weight",
            "randomness": "none",
        },
        "changed": {
            "cell_internal_direction": "weak-cell 2x4 complex direction blended toward v172 teacher",
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
            (__file__, "generate_v176_polue_teacher_direction_transplant.py"),
            (PATHS.code / "generate_v175_spatial_protected_rank_overdrive.py", "generate_v175_spatial_protected_rank_overdrive.py"),
            (PATHS.code / "generate_v174_rank_spatial_intersection_denoising.py", "generate_v174_rank_spatial_intersection_denoising.py"),
            (PATHS.code / "generate_v173_spatial_coherence_tail_denoising.py", "generate_v173_spatial_coherence_tail_denoising.py"),
            (PATHS.code / "generate_v172_rank1_polue_tail_denoising.py", "generate_v172_rank1_polue_tail_denoising.py"),
            (PATHS.code / "generate_v171_full_complex_ad_tail_denoising.py", "generate_v171_full_complex_ad_tail_denoising.py"),
            (PATHS.code / "generate_v170_adaptive_floor_subtraction_denoising.py", "generate_v170_adaptive_floor_subtraction_denoising.py"),
            (PATHS.code / "generate_v169_power_scaled_tail_denoising.py", "generate_v169_power_scaled_tail_denoising.py"),
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
