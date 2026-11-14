#!/usr/bin/env python3
"""Generate v170: adaptive AD floor-subtraction denoising on top of v163.

Hypothesis:
    v169 proved that the released train labels likely contain a weak,
    power-dependent AD tail that should be treated as less reliable than strong
    paths.  v170 makes that denoising more physically explicit: estimate each
    point's collapsed AD noise floor from cells outside the top512, subtract a
    bounded fraction of that floor from weak cells, and protect strong ranked
    paths.

This is still deterministic denoising, not random noise injection.  Compared
with v169's hand-shaped SNR keep curve, v170 behaves more like subtracting an
estimated measurement-noise floor in the AD domain.

Preserved:
    v163 phase source, strong AD components, and 2-pol x 4-UE allocation.  The
    only changed objects are collapsed joint AD support and a bounded point
    level amplitude scale.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from generate_v159_frontier_consensus_support_soup import (
    FLOOR,
    REGIMES,
    collapsed_support_from_output,
    compare_outputs,
    digest,
    normalize_support,
)
from generate_v161_hminus3_tail_distillation import l1_h_change, row_cosine
from generate_v169_power_scaled_tail_denoising import (
    apply_power_scale,
    by_regime_summary,
    support_metrics,
    top_mask,
)
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v170_adaptive_floor_subtraction_denoising"
OUT_DIR = PATHS.cache_dir(170)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(170)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v170_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v170_adaptive_floor_subtraction_target.npy"
WEIGHTS_OUT = OUT_DIR / "v170_adaptive_floor_subtraction_weights.npy"
ACTION_OUT = OUT_DIR / "v170_adaptive_floor_subtraction_actions.npz"
REPORT = REPORT_DIR / "v170_adaptive_floor_subtraction_generate.json"
ZIP_OUT = PATHS.round1_versions / "v170_adaptive_floor_subtraction_denoising_candidate_pending.zip"


def support_cache() -> Path:
    candidates = [
        PATHS.cache_dir(163) / "probe_v163_collapsed_support.npy",
        OUT_DIR / "v163_collapsed_support.npy",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return OUT_DIR / "v163_collapsed_support.npy"


def build_target(
    anchor: np.ndarray,
    regimes: np.ndarray,
    risk: np.ndarray,
    previous_action: np.ndarray,
    family: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    metrics = support_metrics(anchor)
    flat = anchor.reshape(len(anchor), -1)

    regime_scale = np.zeros(len(regimes), dtype=np.float32)
    regime_scale[regimes == "clear"] = 0.10
    regime_scale[regimes == "light"] = 0.16
    regime_scale[regimes == "medium"] = 0.95
    regime_scale[regimes == "heavy"] = 1.45

    tail_gate = np.clip((metrics["tail512_fraction"] - 0.020) / 0.105, 0.0, 1.0)
    entropy_gate = np.clip((metrics["entropy"] - 0.32) / 0.26, 0.0, 1.0)
    top64_loose_gate = np.clip((0.95 - metrics["top64_fraction"]) / 0.46, 0.0, 1.0)
    risk_gate = np.clip((risk - 0.34) / 0.56, 0.0, 1.0).astype(np.float32)
    prior_gate = np.clip(previous_action / 0.72, 0.0, 1.0).astype(np.float32)

    top1_two = family["test_top1_two"].astype(np.float32)
    two_power = family["test_two_power_frac"].astype(np.float32)
    assigned_power = family["test_assigned_power_frac"].astype(np.float32)

    geometry_confidence = np.clip((assigned_power - 0.08) / 0.62, 0.0, 1.0)
    multibounce_protect = (
        1.0
        - 0.30 * np.clip((top1_two - 0.74) / 0.22, 0.0, 1.0)
        - 0.16 * np.clip((two_power - 0.48) / 0.28, 0.0, 1.0)
    ).astype(np.float32)
    multibounce_protect = np.clip(multibounce_protect, 0.52, 1.0)

    base_gate = (
        regime_scale
        * tail_gate
        * np.clip(0.44 + 0.56 * entropy_gate, 0.0, 1.0)
        * np.clip(0.50 + 0.50 * top64_loose_gate, 0.0, 1.0)
        * np.clip(0.50 + 0.30 * risk_gate + 0.20 * prior_gate, 0.0, 1.0)
        * np.clip(0.60 + 0.40 * geometry_confidence, 0.0, 1.0)
    ).astype(np.float32)

    action = np.clip(base_gate * multibounce_protect, 0.0, 1.0).astype(np.float32)
    action[regimes == "clear"] = np.minimum(action[regimes == "clear"], 0.08)
    action[regimes == "light"] = np.minimum(action[regimes == "light"], 0.12)

    mask128 = top_mask(flat, 128)
    mask512 = top_mask(flat, 512)
    mask2048 = top_mask(flat, 2048)
    mask4096 = top_mask(flat, 4096)

    floor = np.maximum(metrics["tail512_floor"], FLOOR).astype(np.float32)
    floor_lambda = (1.75 * action).astype(np.float32)

    rank_multiplier = np.ones(flat.shape, dtype=np.float32)
    rank_multiplier[mask4096] = 0.88
    rank_multiplier[mask2048] = 0.66
    rank_multiplier[mask512] = 0.40
    rank_multiplier[mask128] = 0.0

    min_keep = np.full(flat.shape, 0.025, dtype=np.float32)
    min_keep[mask4096] = 0.09
    min_keep[mask2048] = 0.18
    min_keep[mask512] = 0.50
    min_keep[mask128] = 1.0

    subtracted = flat - floor_lambda[:, None] * floor[:, None] * rank_multiplier
    denoised_flat = np.maximum(subtracted, flat * min_keep).astype(np.float32)
    removed_mass = np.sum(flat - denoised_flat, axis=1).astype(np.float32)
    target = normalize_support(denoised_flat.reshape(anchor.shape))

    lower = np.ones(len(regimes), dtype=np.float32)
    lower[regimes == "clear"] = 0.994
    lower[regimes == "light"] = 0.990
    lower[regimes == "medium"] = 0.945
    lower[regimes == "heavy"] = 0.905
    power_factor = np.clip(1.0 - 0.54 * removed_mass, lower, 1.0).astype(np.float32)

    diagnostics = {
        **metrics,
        "risk": risk.astype(np.float32),
        "previous_action": previous_action.astype(np.float32),
        "regime_scale": regime_scale,
        "tail_gate": tail_gate.astype(np.float32),
        "entropy_gate": entropy_gate.astype(np.float32),
        "top64_loose_gate": top64_loose_gate.astype(np.float32),
        "risk_gate": risk_gate,
        "prior_gate": prior_gate,
        "geometry_confidence": geometry_confidence.astype(np.float32),
        "multibounce_protect": multibounce_protect,
        "action": action,
        "floor_lambda": floor_lambda,
        "removed_mass": removed_mass,
        "power_factor": power_factor,
        "top1_two": top1_two,
        "two_power_frac": two_power,
        "assigned_power_frac": assigned_power,
        "floor": floor,
    }
    return target, power_factor, diagnostics


def main() -> None:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked_test = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "test_blocked"
    ].astype(np.float32)
    regimes = regime_of(blocked_test)
    risk = np.load(PATHS.cache_dir(140) / "v140_geodesic_risk.npy").astype(np.float32)

    anchor = collapsed_support_from_output(
        PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy",
        pos_test,
        support_cache(),
    )
    v163_actions = np.load(
        PATHS.cache_dir(163) / "v163_validated_momentum_overdrive_actions.npz"
    )
    previous_action = v163_actions["action"].astype(np.float32)
    family = np.load(PATHS.shared_features / "v166_path_family_predictability_predictions.npz")

    target, power_factor, diag = build_target(
        anchor, regimes, risk, previous_action, family
    )
    np.save(TARGET_OUT, target)

    action = diag["action"].astype(np.float32)
    max_weight = np.clip(0.72 + 0.26 * action, 0.72, 0.98).astype(np.float32)
    weights = np.empty((len(action), 8), dtype=np.float32)
    weights[:, 0] = max_weight
    weights[:, 1:] = (1.0 - max_weight[:, None]) / 7.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    np.save(WEIGHTS_OUT, weights)

    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        weights=weights,
        **diag,
    )

    args = argparse.Namespace(
        base=str(PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy"),
        output=str(OUTPUT),
        beta=140.3,
        gamma=0.10,
        eta="clear:0.40,light:0.54,medium:1.02,heavy:1.28",
        min_gate=0.05,
        max_gate=2.96,
        conf_center=0.32,
        conf_width=0.32,
        entropy_low=0.04,
        entropy_high=0.94,
        gain_clip="0.16,3.72",
        top_gain_clip="0.58,1.88",
        top_components=512,
        target_floor=FLOOR,
        project_batch=20,
    )
    projection_report = projection(target, weights, regimes, pos_test, args)
    apply_power_scale(OUTPUT, power_factor)

    comparison_to_v163 = compare_outputs(
        OUTPUT, PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v169 = compare_outputs(
        OUTPUT, PATHS.cache_dir(169) / "v169_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v140 = compare_outputs(
        OUTPUT, PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", regimes
    )
    if comparison_to_v163["shape"] != [500, 256, 4, 192] or comparison_to_v163["dtype"] != "complex64":
        raise ValueError(
            f"invalid output layout: {comparison_to_v163['shape']} {comparison_to_v163['dtype']}"
        )
    if not comparison_to_v163["finite"]:
        raise ValueError("v170 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v169 validated weak AD tail denoising. v170 replaces the fixed SNR "
            "keep curve with adaptive floor subtraction: estimate each point's "
            "AD noise floor outside top512, subtract a bounded floor amount from "
            "weak cells, protect strong paths, and shrink total power only by a "
            "fraction of removed tail mass."
        ),
        "new_information_used": {
            "v169_online": "0.503428 confirms weak-tail reliability denoising improves v163",
            "v168": "AD tail has circular-complex-like behavior and power-dependent floor",
            "v166": "path-family signal protects likely real multi-bounce components",
            "v163": "phase/pol-UE base and online-validated support momentum",
        },
        "excluded": {
            "random_noise_injection": True,
            "away_from_negative_common_subspace": True,
            "broad_sionna_delay_controller": True,
            "full_phase_relearning": True,
            "exact_zero_deletion": True,
        },
        "parameters": {
            "support_floor": "tail512_floor = mean normalized AD support excluding top512 cells",
            "action": "regime/tail/entropy/top64/risk/prior/geometry gates with multi-bounce protection",
            "regime_scale": {"clear": 0.10, "light": 0.16, "medium": 0.95, "heavy": 1.45},
            "easy_caps": {"clear": 0.08, "light": 0.12},
            "floor_lambda": "1.75 * action",
            "rank_multiplier": {
                "top128": 0.0,
                "top512": 0.40,
                "top2048": 0.66,
                "top4096": 0.88,
                "tail": 1.0,
            },
            "min_keep": {
                "top128": 1.0,
                "top512": 0.50,
                "top2048": 0.18,
                "top4096": 0.09,
                "tail": 0.025,
            },
            "power_factor": "clip(1 - 0.54*removed_tail_mass, regime lower bound, 1.0)",
            "projection_args": {
                key: (PATHS.relative(value) if key in {"base", "output"} else value)
                for key, value in vars(args).items()
            },
        },
        "diagnostics": {
            "overall": {
                key: summarize(value)
                for key, value in diag.items()
                if isinstance(value, np.ndarray) and value.ndim == 1
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "action": action,
                    "floor_lambda": diag["floor_lambda"],
                    "removed_mass": diag["removed_mass"],
                    "power_factor": power_factor,
                    "tail512_fraction": diag["tail512_fraction"],
                    "tail128_fraction": diag["tail128_fraction"],
                    "top64_fraction": diag["top64_fraction"],
                    "entropy": diag["entropy"],
                    "multibounce_protect": diag["multibounce_protect"],
                },
            ),
            "active_fraction": float(np.mean(action > 1e-6)),
            "power_shrink_fraction": float(np.mean(power_factor < 0.999999)),
        },
        "target_diagnostics": {
            "target_to_v163_support": summarize(row_cosine(target, anchor)),
            "target_h_l1_change_to_v163": summarize(l1_h_change(target, anchor)),
        },
        "projection": projection_report,
        "comparison_to_v163": comparison_to_v163,
        "comparison_to_v169": comparison_to_v169,
        "comparison_to_v140": comparison_to_v140,
        "preserved": {
            "base_complex": "v163",
            "strong_paths": "top128 joint AD cells are never attenuated in target construction",
            "phase": "inherits v163 phase through positive AD amplitude projection",
            "pol_ue_allocation": "projection scales each collapsed H/V/delay component uniformly over pol/UE",
        },
        "changed": {
            "collapsed_joint_ad_support": "adaptive AD floor subtraction and renormalization",
            "total_power": "bounded deterministic shrink from removed tail mass",
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET_OUT),
            "weights": PATHS.relative(WEIGHTS_OUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source, name in (
            (__file__, "generate_v170_adaptive_floor_subtraction_denoising.py"),
            (PATHS.code / "generate_v169_power_scaled_tail_denoising.py", "generate_v169_power_scaled_tail_denoising.py"),
            (PATHS.code / "analyze_v168_channel_awgn_power_audit.py", "analyze_v168_channel_awgn_power_audit.py"),
            (PATHS.code / "generate_v163_validated_momentum_overdrive.py", "generate_v163_validated_momentum_overdrive.py"),
            (PATHS.code / "generate_v159_frontier_consensus_support_soup.py", "generate_v159_frontier_consensus_support_soup.py"),
            (PATHS.code / "generate_v98_path_ensemble_graph_attention_joint_ad.py", "generate_v98_path_ensemble_graph_attention_joint_ad.py"),
            (PATHS.code / "generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
            (PATHS.code / "generate_v161_hminus3_tail_distillation.py", "generate_v161_hminus3_tail_distillation.py"),
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
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
