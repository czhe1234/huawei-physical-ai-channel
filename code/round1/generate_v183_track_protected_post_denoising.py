#!/usr/bin/env python3
"""Generate v183: track-protected post-track AD tail denoising.

v181 is the current online best, but analysis shows the v178-v181 component
track passes also increased heavy-region AD tail entropy/floor relative to
v171.  This candidate uses a new order of operations:

1. keep v181 as the complex backbone;
2. protect strong AD cells and v181-tracked component cells;
3. protect high-confidence two-bounce/path-family regions from v166;
4. shrink only weak, non-track AD coefficients, with bounded total-power shrink.

The hypothesis is that v171 removed weak noisy tails before the track branch,
while the later support moves can re-amplify non-track weak cells.  v183 tests
whether a track-aware second denoising pass can keep the v181 propagation
support while removing residual noisy speckles.
"""

from __future__ import annotations

import json
import math
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
from generate_v159_frontier_consensus_support_soup import REGIMES, digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v183_track_protected_post_denoising"
OUT_DIR = PATHS.cache_dir(183)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(183)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v183_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v183_track_protected_post_denoising_actions.npz"
REPORT = REPORT_DIR / "v183_track_protected_post_denoising_generate.json"
ZIP_OUT = PATHS.round1_versions / "v183_track_protected_post_denoising_candidate_pending.zip"

BASE = PATHS.cache_dir(181) / "v181_Round1_Test_Channel.npy"
V181_ACTIONS = PATHS.cache_dir(181) / "v181_multihop_track_field_actions.npz"
V166_PREDS = PATHS.shared_features / "v166_path_family_predictability_predictions.npz"

CELL_COUNT = MH * MV * S_SC
COMP_PER_CELL = MP * N_ANT
FLOOR = 1e-12


def support_metrics(cell: np.ndarray) -> dict[str, np.ndarray]:
    total = np.maximum(cell.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
    support = cell / total[:, None, None, None]
    flat = support.reshape(len(cell), -1).astype(np.float32)
    sorted_flat = np.sort(flat, axis=1)
    entropy = -np.sum(flat * np.log(np.maximum(flat, FLOOR)), axis=1) / math.log(CELL_COUNT)
    return {
        "support": support.astype(np.float32),
        "flat": flat,
        "cell_total_power": total,
        "top64_fraction": sorted_flat[:, -64:].sum(axis=1).astype(np.float32),
        "top128_fraction": sorted_flat[:, -128:].sum(axis=1).astype(np.float32),
        "top512_fraction": sorted_flat[:, -512:].sum(axis=1).astype(np.float32),
        "tail512_fraction": (sorted_flat[:, :-512].mean(axis=1) * CELL_COUNT).astype(np.float32),
        "tail512_floor": sorted_flat[:, :-512].mean(axis=1).astype(np.float32),
        "entropy": entropy.astype(np.float32),
    }


def add_track_protection(mask: np.ndarray, cells: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Mark v181 tracked component neighborhoods as protected AD cells."""
    protected = mask.copy()
    for i in range(len(protected)):
        for slot in range(cells.shape[1]):
            w = float(weights[i, slot])
            if w <= 0:
                break
            h0, v0, d0 = cells[i, slot]
            if h0 < 0:
                break
            for dh in (-1, 0, 1):
                hh = (int(h0) + dh) % MH
                for dv in (-1, 0, 1):
                    vv = int(v0) + dv
                    if vv < 0 or vv >= MV:
                        continue
                    for dd in range(-4, 5):
                        tau = int(d0) + dd
                        if 0 <= tau < S_SC:
                            protected[i, hh, vv, tau] = True
    return protected


def action_policy(
    regimes: np.ndarray,
    metrics: dict[str, np.ndarray],
    track_conf: np.ndarray,
    route_evidence: np.ndarray,
    net_alpha: np.ndarray,
    top1_two: np.ndarray,
) -> dict[str, np.ndarray]:
    tail = metrics["tail512_fraction"]
    entropy = metrics["entropy"]

    tail_gate = np.clip((tail - 0.040) / 0.070, 0.0, 1.0).astype(np.float32)
    ent_gate = np.clip((entropy - 0.500) / 0.140, 0.0, 1.0).astype(np.float32)
    track_gate = np.clip((track_conf - 0.54) / 0.24, 0.0, 1.0).astype(np.float32)
    route_gate = 0.72 + 0.28 * np.clip((route_evidence - 0.045) / 0.16, 0.0, 1.0).astype(np.float32)
    active_gate = 0.62 + 0.38 * np.clip((net_alpha + 0.01) / 0.28, 0.0, 1.0).astype(np.float32)

    # v166 says high top1_two regions are multi-bounce rich.  In those points,
    # denoise less aggressively because weak non-LOS tails are more likely real.
    two_bounce_protect = np.clip((top1_two - 0.45) / 0.45, 0.0, 1.0).astype(np.float32)
    multibounce_keep = 1.0 - 0.36 * two_bounce_protect

    scale = np.zeros(len(regimes), dtype=np.float32)
    scale[regimes == "medium"] = 0.42
    scale[regimes == "heavy"] = 0.92
    # Light points are normally preserved, but a tiny action tests whether
    # obvious weak-tail speckle cleanup has value outside medium/heavy.
    scale[regimes == "light"] = 0.055
    scale[regimes == "clear"] = 0.0

    action = scale * tail_gate * (0.45 + 0.55 * ent_gate) * track_gate * route_gate * active_gate * multibounce_keep
    action = np.clip(action, 0.0, 1.0).astype(np.float32)
    return {
        "tail_gate": tail_gate,
        "entropy_gate": ent_gate,
        "track_gate": track_gate,
        "route_gate": route_gate.astype(np.float32),
        "active_gate": active_gate.astype(np.float32),
        "two_bounce_protect": two_bounce_protect,
        "multibounce_keep": multibounce_keep,
        "action": action,
    }


def denoise_batch(
    ad: np.ndarray,
    regimes: np.ndarray,
    track_cells: np.ndarray,
    track_weights: np.ndarray,
    track_conf: np.ndarray,
    route_evidence: np.ndarray,
    net_alpha: np.ndarray,
    top1_two: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    coeff_power = (np.abs(ad) ** 2).astype(np.float32)
    cell = coeff_power.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    metrics = support_metrics(cell)
    flat = metrics["flat"]

    mask64 = top_mask(flat, 64).reshape(len(ad), MH, MV, S_SC)
    mask128 = top_mask(flat, 128).reshape(len(ad), MH, MV, S_SC)
    mask512 = top_mask(flat, 512).reshape(len(ad), MH, MV, S_SC)
    mask2048 = top_mask(flat, 2048).reshape(len(ad), MH, MV, S_SC)
    mask4096 = top_mask(flat, 4096).reshape(len(ad), MH, MV, S_SC)
    track_mask = add_track_protection(mask128, track_cells, track_weights)

    actions = action_policy(regimes, metrics, track_conf, route_evidence, net_alpha, top1_two)
    action = actions["action"]

    rank_multiplier = np.ones(flat.shape, dtype=np.float32)
    rank_multiplier[mask4096.reshape(len(ad), -1)] = 0.92
    rank_multiplier[mask2048.reshape(len(ad), -1)] = 0.66
    rank_multiplier[mask512.reshape(len(ad), -1)] = 0.36
    rank_multiplier[mask128.reshape(len(ad), -1)] = 0.0
    rank_multiplier[track_mask.reshape(len(ad), -1)] = 0.0

    min_keep = np.full(flat.shape, 0.020, dtype=np.float32)
    min_keep[mask4096.reshape(len(ad), -1)] = 0.060
    min_keep[mask2048.reshape(len(ad), -1)] = 0.160
    min_keep[mask512.reshape(len(ad), -1)] = 0.580
    min_keep[mask128.reshape(len(ad), -1)] = 1.0
    min_keep[mask64.reshape(len(ad), -1)] = 1.0
    min_keep[track_mask.reshape(len(ad), -1)] = 1.0

    floor_cell_power = (np.maximum(metrics["tail512_floor"], FLOOR) * metrics["cell_total_power"]).astype(np.float32)
    coeff_floor = (floor_cell_power / float(COMP_PER_CELL)).astype(np.float32)
    coeff_lambda = (3.85 * action).astype(np.float32)

    rank = rank_multiplier.reshape(len(ad), 1, MH, MV, 1, S_SC)
    keep = min_keep.reshape(len(ad), 1, MH, MV, 1, S_SC)
    subtract = coeff_lambda[:, None, None, None, None, None] * coeff_floor[:, None, None, None, None, None] * rank
    denoised_power = np.maximum(coeff_power - subtract, coeff_power * keep)

    total0 = np.maximum(coeff_power.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
    total_direct = np.maximum(denoised_power.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
    removed_fraction = np.clip(1.0 - total_direct / total0, 0.0, 1.0).astype(np.float32)

    lower = np.ones(len(ad), dtype=np.float32)
    lower[regimes == "light"] = 0.996
    lower[regimes == "medium"] = 0.988
    lower[regimes == "heavy"] = 0.974
    target_power_factor = np.clip(1.0 - 0.44 * removed_fraction, lower, 1.0).astype(np.float32)

    gain = np.sqrt(denoised_power / np.maximum(coeff_power, 1e-30)).astype(np.float32)
    ad1 = (ad * gain).astype(np.complex64)
    total_after_gain = np.maximum(
        np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64),
        TINY,
    ).astype(np.float32)
    scale = np.sqrt(target_power_factor * total0 / total_after_gain).astype(np.float32)
    ad1 *= scale[:, None, None, None, None, None]

    cell1 = (np.abs(ad1) ** 2).astype(np.float32).sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    support1 = cell1 / np.maximum(cell1.sum(axis=(1, 2, 3), keepdims=True), TINY)
    support_l1 = np.sum(np.abs(support1 - metrics["support"]), axis=(1, 2, 3)).astype(np.float32)

    diag = {
        **actions,
        "top64_fraction": metrics["top64_fraction"],
        "top128_fraction": metrics["top128_fraction"],
        "top512_fraction": metrics["top512_fraction"],
        "tail512_fraction": metrics["tail512_fraction"],
        "entropy": metrics["entropy"],
        "floor_cell_power": floor_cell_power,
        "coeff_lambda": coeff_lambda,
        "track_protected_cells": track_mask.reshape(len(ad), -1).sum(axis=1).astype(np.float32),
        "removed_fraction_direct": removed_fraction,
        "target_power_factor": target_power_factor,
        "actual_power_factor": (
            np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64)
            / np.maximum(total0, TINY)
        ).astype(np.float32),
        "support_l1_to_base": support_l1,
    }
    return ad1.astype(np.complex64), diag


def compare_output_to(output_path, base_path, regimes: np.ndarray) -> dict[str, object]:
    output = np.load(output_path, mmap_mode="r")
    base = np.load(base_path, mmap_mode="r")
    numerator = np.zeros(len(output), dtype=np.float64)
    denominator = np.zeros(len(output), dtype=np.float64)
    out_power = np.zeros(len(output), dtype=np.float64)
    base_power = np.zeros(len(output), dtype=np.float64)
    finite = True
    for start in range(0, len(output), 20):
        end = min(start + 20, len(output))
        current = np.asarray(output[start:end], dtype=np.complex64)
        reference = np.asarray(base[start:end], dtype=np.complex64)
        numerator[start:end] = np.sum(np.abs(current - reference) ** 2, axis=(1, 2, 3), dtype=np.float64)
        denominator[start:end] = np.sum(np.abs(reference) ** 2, axis=(1, 2, 3), dtype=np.float64)
        out_power[start:end] = np.sum(np.abs(current) ** 2, axis=(1, 2, 3), dtype=np.float64)
        base_power[start:end] = denominator[start:end]
        finite = finite and bool(np.isfinite(current.real).all() and np.isfinite(current.imag).all())
    point_nmse = numerator / np.maximum(denominator, TINY)
    return {
        "finite": finite,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "nmse": float(numerator.sum() / max(denominator.sum(), TINY)),
        "nmse_by_regime": {
            regime: float(numerator[regimes == regime].sum() / max(denominator[regimes == regime].sum(), TINY))
            for regime in REGIMES
            if np.any(regimes == regime)
        },
        "point_nmse": summarize(point_nmse),
        "power_ratio": summarize(out_power / np.maximum(base_power, TINY)),
    }


def generate() -> dict[str, object]:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    actions = np.load(V181_ACTIONS)
    top1_two = np.load(V166_PREDS)["test_top1_two"].astype(np.float32)

    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    beta_s = beta_subcarriers(140.3, 0.10)
    distance = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)

    diag_lists: dict[str, list[np.ndarray]] = {}

    for start in range(0, len(source), 20):
        end = min(start + 20, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        derotated = h0 * np.exp(
            1j * distance[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        ad0 = ad_transform(derotated)
        ad1, diag = denoise_batch(
            ad0,
            regimes[start:end],
            actions["track_cells"][start:end].astype(np.int16),
            actions["track_weights"][start:end].astype(np.float32),
            actions["track_conf"][start:end].astype(np.float32),
            actions["mean_route_evidence"][start:end].astype(np.float32),
            actions["net_alpha"][start:end].astype(np.float32),
            top1_two[start:end],
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
            diag_lists.setdefault(key, []).append(value.astype(np.float32))
        print(f"v183 post denoising {end}/{len(source)}", flush=True)

    diag_all = {key: np.concatenate(value).astype(np.float32) for key, value in diag_lists.items()}
    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        top1_two=top1_two,
        v181_track_conf=actions["track_conf"].astype(np.float32),
        v181_mean_route_evidence=actions["mean_route_evidence"].astype(np.float32),
        v181_net_alpha=actions["net_alpha"].astype(np.float32),
        **diag_all,
    )

    comparison_to_v181 = compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v180 = compare_output_to(OUTPUT, PATHS.cache_dir(180) / "v180_Round1_Test_Channel.npy", regimes)
    comparison_to_v173 = compare_output_to(OUTPUT, PATHS.cache_dir(173) / "v173_Round1_Test_Channel.npy", regimes)
    if comparison_to_v181["shape"] != [500, 256, 4, 192] or comparison_to_v181["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v181['shape']} {comparison_to_v181['dtype']}")
    if not comparison_to_v181["finite"]:
        raise ValueError("v183 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v178-v181 improved support by tracking components, but also raised "
            "heavy-region AD tail floor/entropy.  v183 keeps v181 tracked cells "
            "and strong cells, then denoises only weak non-track AD coefficients."
        ),
        "new_information_used": {
            "current_best": "v181 online 0.504284",
            "tail_audit": "v181 heavy tail512/entropy is higher than v171 after multiple support moves",
            "track_protection": "v181 track_cells/track_weights protect coherent propagation components",
            "path_family_protection": "v166 top1_two reduces denoising in predicted multibounce-rich regions",
            "guardrails": "no global phase regression; no train-truth peak prior; bounded total-power shrink",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "protected": "top64/top128 + v181 track-cell neighborhoods",
            "regime_action": "clear=0, light tiny, medium moderate, heavy strong",
            "coeff_lambda": "3.85 * action",
            "power_policy": "bounded shrink: light>=0.996, medium>=0.988, heavy>=0.974",
            "batch": 20,
        },
        "diagnostics": {
            "overall": {key: summarize(value) for key, value in diag_all.items()},
            "by_regime": by_regime_summary(regimes, {"action": diag_all["action"]}),
            "support_l1_by_regime": by_regime_summary(regimes, {"support_l1_to_base": diag_all["support_l1_to_base"]}),
            "power_factor_by_regime": by_regime_summary(regimes, {"actual_power_factor": diag_all["actual_power_factor"]}),
        },
        "comparison_to_v181": comparison_to_v181,
        "comparison_to_v180": comparison_to_v180,
        "comparison_to_v173": comparison_to_v173,
        "preserved": {
            "base_complex": "v181 is the complex backbone",
            "phase_policy": "no absolute phase transport or global phase regression",
            "tracked_components": "track cells are excluded from shrink",
        },
        "changed": {
            "weak_nontrack_ad_tail": "coefficients outside strong/track cells are floor-subtracted",
            "point_total_power": "may shrink slightly when weak-tail mass is removed",
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }

    del output
    zip_members = ["Round1_Test_Channel.npy"]
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(OUTPUT, "Round1_Test_Channel.npy")
        for src in [
            PATHS.code / "generate_v183_track_protected_post_denoising.py",
            PATHS.code / "generate_v87_easy_field_projection.py",
            PATHS.code / "generate_v171_full_complex_ad_tail_denoising.py",
            PATHS.code / "generate_v169_power_scaled_tail_denoising.py",
            PATHS.code / "generate_v159_frontier_consensus_support_soup.py",
            PATHS.code / "powerfield_common.py",
            PATHS.code_root / "project_paths.py",
        ]:
            zf.write(src, src.name)
            zip_members.append(src.name)

    checked = np.load(OUTPUT, mmap_mode="r")
    finite = True
    for start in range(0, len(checked), 20):
        end = min(start + 20, len(checked))
        block = np.asarray(checked[start:end], dtype=np.complex64)
        finite = finite and bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
    report["integrity"] = {
        "shape": list(checked.shape),
        "dtype": str(checked.dtype),
        "finite": finite,
        "zip_crc": True,
        "zip_members": zip_members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    generate()
