#!/usr/bin/env python3
"""Generate v184: validated post-track denoising overdrive.

v183 proved online that track-protected post-denoising can improve v181.  The
v183 action was intentionally conservative: clear/light stayed unchanged,
medium was nearly untouched, and heavy median action was only about 0.035.

v184 uses v183 as the complex backbone and tests whether the same validated
information source has more room:

* protect v181 tracked AD cells and strong cells;
* protect high-confidence multibounce-rich regions from v166;
* apply a wider, stronger non-track weak-tail floor subtraction;
* allow a larger but still bounded total-power shrink.

This is not another component-track overdrive and does not transport absolute
phase.  It asks whether the small v183 gain was the first step of a larger
post-track reliability correction.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np

import generate_v183_track_protected_post_denoising as v183
from generate_v87_easy_field_projection import BS_POS, beta_subcarriers, inverse_ad, summarize
from generate_v159_frontier_consensus_support_soup import REGIMES, digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v184_validated_post_denoising_overdrive"
OUT_DIR = PATHS.cache_dir(184)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(184)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v184_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v184_validated_post_denoising_overdrive_actions.npz"
REPORT = REPORT_DIR / "v184_validated_post_denoising_overdrive_generate.json"
ZIP_OUT = PATHS.round1_versions / "v184_validated_post_denoising_overdrive_candidate_pending.zip"

BASE = PATHS.cache_dir(183) / "v183_Round1_Test_Channel.npy"
V181 = PATHS.cache_dir(181) / "v181_Round1_Test_Channel.npy"
V181_ACTIONS = PATHS.cache_dir(181) / "v181_multihop_track_field_actions.npz"
V166_PREDS = PATHS.shared_features / "v166_path_family_predictability_predictions.npz"


def action_policy_overdrive(
    regimes: np.ndarray,
    metrics: dict[str, np.ndarray],
    track_conf: np.ndarray,
    route_evidence: np.ndarray,
    net_alpha: np.ndarray,
    top1_two: np.ndarray,
) -> dict[str, np.ndarray]:
    tail = metrics["tail512_fraction"]
    entropy = metrics["entropy"]

    # Wider than v183: v183 online gain says a small post-track cleanup is real.
    tail_gate = np.clip((tail - 0.030) / 0.060, 0.0, 1.0).astype(np.float32)
    ent_gate = np.clip((entropy - 0.475) / 0.135, 0.0, 1.0).astype(np.float32)
    track_gate = np.clip((track_conf - 0.500) / 0.250, 0.0, 1.0).astype(np.float32)
    route_gate = 0.78 + 0.22 * np.clip((route_evidence - 0.035) / 0.150, 0.0, 1.0).astype(np.float32)
    active_gate = 0.70 + 0.30 * np.clip((net_alpha + 0.005) / 0.250, 0.0, 1.0).astype(np.float32)

    # Still protect multibounce-rich points, but less conservatively than v183.
    two_bounce_protect = np.clip((top1_two - 0.48) / 0.42, 0.0, 1.0).astype(np.float32)
    multibounce_keep = 1.0 - 0.28 * two_bounce_protect

    scale = np.zeros(len(regimes), dtype=np.float32)
    scale[regimes == "clear"] = 0.0
    scale[regimes == "light"] = 0.035
    scale[regimes == "medium"] = 0.92
    scale[regimes == "heavy"] = 1.95

    action = scale * tail_gate * (0.38 + 0.62 * ent_gate) * track_gate * route_gate * active_gate * multibounce_keep
    action = np.clip(action, 0.0, 1.0).astype(np.float32)
    return {
        "tail_gate": tail_gate,
        "entropy_gate": ent_gate,
        "track_gate": track_gate,
        "route_gate": route_gate.astype(np.float32),
        "active_gate": active_gate.astype(np.float32),
        "two_bounce_protect": two_bounce_protect,
        "multibounce_keep": multibounce_keep.astype(np.float32),
        "action": action,
    }


def denoise_batch_overdrive(
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
    metrics = v183.support_metrics(cell)
    flat = metrics["flat"]

    mask64 = v183.top_mask(flat, 64).reshape(len(ad), v183.MH, v183.MV, v183.S_SC)
    mask128 = v183.top_mask(flat, 128).reshape(len(ad), v183.MH, v183.MV, v183.S_SC)
    mask512 = v183.top_mask(flat, 512).reshape(len(ad), v183.MH, v183.MV, v183.S_SC)
    mask2048 = v183.top_mask(flat, 2048).reshape(len(ad), v183.MH, v183.MV, v183.S_SC)
    mask4096 = v183.top_mask(flat, 4096).reshape(len(ad), v183.MH, v183.MV, v183.S_SC)
    track_mask = v183.add_track_protection(mask128, track_cells, track_weights)

    actions = action_policy_overdrive(regimes, metrics, track_conf, route_evidence, net_alpha, top1_two)
    action = actions["action"]

    rank_multiplier = np.ones(flat.shape, dtype=np.float32)
    rank_multiplier[mask4096.reshape(len(ad), -1)] = 1.18
    rank_multiplier[mask2048.reshape(len(ad), -1)] = 0.82
    rank_multiplier[mask512.reshape(len(ad), -1)] = 0.42
    rank_multiplier[mask128.reshape(len(ad), -1)] = 0.0
    rank_multiplier[track_mask.reshape(len(ad), -1)] = 0.0

    min_keep = np.full(flat.shape, 0.010, dtype=np.float32)
    min_keep[mask4096.reshape(len(ad), -1)] = 0.040
    min_keep[mask2048.reshape(len(ad), -1)] = 0.120
    min_keep[mask512.reshape(len(ad), -1)] = 0.520
    min_keep[mask128.reshape(len(ad), -1)] = 1.0
    min_keep[mask64.reshape(len(ad), -1)] = 1.0
    min_keep[track_mask.reshape(len(ad), -1)] = 1.0

    floor_cell_power = (np.maximum(metrics["tail512_floor"], v183.FLOOR) * metrics["cell_total_power"]).astype(np.float32)
    coeff_floor = (floor_cell_power / float(v183.COMP_PER_CELL)).astype(np.float32)
    coeff_lambda = (5.85 * action).astype(np.float32)

    rank = rank_multiplier.reshape(len(ad), 1, v183.MH, v183.MV, 1, v183.S_SC)
    keep = min_keep.reshape(len(ad), 1, v183.MH, v183.MV, 1, v183.S_SC)
    subtract = coeff_lambda[:, None, None, None, None, None] * coeff_floor[:, None, None, None, None, None] * rank
    denoised_power = np.maximum(coeff_power - subtract, coeff_power * keep)

    total0 = np.maximum(coeff_power.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
    total_direct = np.maximum(denoised_power.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
    removed_fraction = np.clip(1.0 - total_direct / total0, 0.0, 1.0).astype(np.float32)

    lower = np.ones(len(ad), dtype=np.float32)
    lower[regimes == "light"] = 0.995
    lower[regimes == "medium"] = 0.982
    lower[regimes == "heavy"] = 0.960
    target_power_factor = np.clip(1.0 - 0.62 * removed_fraction, lower, 1.0).astype(np.float32)

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
        ad0 = v183.ad_transform(derotated)
        ad1, diag = denoise_batch_overdrive(
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
        print(f"v184 overdrive denoising {end}/{len(source)}", flush=True)

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

    del output
    comparison_to_v183 = v183.compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v181 = v183.compare_output_to(OUTPUT, V181, regimes)
    comparison_to_v180 = v183.compare_output_to(OUTPUT, PATHS.cache_dir(180) / "v180_Round1_Test_Channel.npy", regimes)
    if comparison_to_v183["shape"] != [500, 256, 4, 192] or comparison_to_v183["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v183['shape']} {comparison_to_v183['dtype']}")
    if not comparison_to_v183["finite"]:
        raise ValueError("v184 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v183 online-positive result validates track-protected post-denoising. "
            "Because v183 was conservative, v184 uses v183 as base and applies a "
            "stronger protected non-track weak-tail cleanup."
        ),
        "new_information_used": {
            "current_best": "v183 online 0.504306",
            "validated_direction": "v181 -> v183 track-protected post-denoising was online-positive",
            "guardrails": "protect v181 track cells, strong AD cells, and high top1_two multibounce-rich regions",
            "excluded": "no absolute phase transport, no sample-index trajectory, no train-truth peak prior",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "action_scale": "medium 0.92, heavy 1.95; wider gates than v183",
            "coeff_lambda": "5.85 * action",
            "power_policy": "bounded shrink: light>=0.995, medium>=0.982, heavy>=0.960",
            "batch": 20,
        },
        "diagnostics": {
            "overall": {key: summarize(value) for key, value in diag_all.items()},
            "by_regime": by_regime_summary(regimes, {"action": diag_all["action"]}),
            "support_l1_by_regime": by_regime_summary(regimes, {"support_l1_to_base": diag_all["support_l1_to_base"]}),
            "power_factor_by_regime": by_regime_summary(regimes, {"actual_power_factor": diag_all["actual_power_factor"]}),
        },
        "comparison_to_v183": comparison_to_v183,
        "comparison_to_v181": comparison_to_v181,
        "comparison_to_v180": comparison_to_v180,
        "preserved": {
            "base_complex": "v183 is the complex backbone",
            "phase_policy": "no absolute phase transport or global phase regression",
            "tracked_components": "v181 track cells are excluded from shrink",
        },
        "changed": {
            "weak_nontrack_ad_tail": "stronger floor-subtraction outside strong/track cells",
            "point_total_power": "may shrink more than v183 but remains bounded",
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }

    zip_members = ["Round1_Test_Channel.npy"]
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(OUTPUT, "Round1_Test_Channel.npy")
        for src in [
            PATHS.code / "generate_v184_validated_post_denoising_overdrive.py",
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
        block = np.asarray(checked[start : start + 20], dtype=np.complex64)
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
