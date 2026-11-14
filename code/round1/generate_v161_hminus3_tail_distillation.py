#!/usr/bin/env python3
"""Generate v161: h=-3 tail distillation from the rejected v139 overdrive.

Hypothesis:
    v140 is the current online best, but v139 was never submitted; it was
    rejected because it looked too aggressive locally.  Output-support
    diagnostics show a more specific fact: the v139-v140 heavy tail is mostly
    aligned with the confirmed online-positive v138->v140 direction, whereas
    the later v160 frontier consensus is mostly anti-aligned in actual output
    support.  v161 therefore does not revive v139 wholesale.  It distills only
    the high-risk, same-direction part of the v139 tail and applies it as a
    bounded support-only projection on top of the v140 complex output.

Preserved:
    v140 complex phase, point total power, clear/light points, and pol/UE
    allocation.  The only changed object is the collapsed joint AD support for
    selected medium/heavy points.
"""

from __future__ import annotations

import argparse
import json
import zipfile

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
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v161_hminus3_tail_distillation"
OUT_DIR = PATHS.cache_dir(161)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(161)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v161_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v161_hminus3_tail_distillation_target.npy"
WEIGHTS_OUT = OUT_DIR / "v161_hminus3_tail_distillation_weights.npy"
ACTION_OUT = OUT_DIR / "v161_hminus3_tail_distillation_actions.npz"
REPORT = REPORT_DIR / "v161_hminus3_tail_distillation_generate.json"
ZIP_OUT = PATHS.round1_versions / "v161_hminus3_tail_distillation_candidate_pending.zip"


def support_cache(version: int) -> str:
    # A previous diagnostic may already have collapsed these immutable outputs.
    probe = OUT_DIR / f"probe_v{version}_collapsed_support.npy"
    if probe.is_file():
        return str(probe)
    return str(OUT_DIR / f"v161_v{version}_collapsed_support.npy")


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a.reshape(a.shape[0], -1).astype(np.float64)
    bb = b.reshape(b.shape[0], -1).astype(np.float64)
    denom = np.sqrt(np.sum(aa * aa, axis=1) * np.sum(bb * bb, axis=1)) + 1e-30
    return (np.sum(aa * bb, axis=1) / denom).astype(np.float32)


def weighted_log_direction(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a clipped, support-weighted log direction from source to target."""
    direction = np.log(np.maximum(target, FLOOR)) - np.log(np.maximum(source, FLOOR))
    direction = np.clip(direction, -3.0, 3.0).astype(np.float32)
    direction *= np.sqrt(np.maximum(source, FLOOR)).astype(np.float32)
    direction -= direction.reshape(direction.shape[0], -1).mean(axis=1)[
        :, None, None, None
    ].astype(np.float32)
    return direction.astype(np.float32)


def l1_h_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ah = a.sum(axis=(2, 3))
    bh = b.sum(axis=(2, 3))
    return np.sum(np.abs(ah - bh), axis=1).astype(np.float32)


def main() -> None:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked_test = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "test_blocked"
    ].astype(np.float32)
    regimes = regime_of(blocked_test)
    risk = np.load(PATHS.cache_dir(140) / "v140_geodesic_risk.npy").astype(np.float32)

    support_paths = {
        138: PATHS.cache_dir(138) / "v138_Round1_Test_Channel.npy",
        139: PATHS.cache_dir(139) / "v139_Round1_Test_Channel.npy",
        140: PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy",
        160: PATHS.cache_dir(160) / "v160_Round1_Test_Channel.npy",
    }
    supports = {
        version: collapsed_support_from_output(
            path, pos_test, PATHS.resolve(support_cache(version))
        )
        for version, path in support_paths.items()
    }
    anchor = supports[140]

    proven_direction = weighted_log_direction(supports[138], supports[140])
    tail_direction = weighted_log_direction(supports[140], supports[139])
    frontier_direction = weighted_log_direction(supports[140], supports[160])

    tail_alignment = row_cosine(tail_direction, proven_direction)
    frontier_alignment = row_cosine(frontier_direction, proven_direction)
    tail_frontier_alignment = row_cosine(tail_direction, frontier_direction)
    tail_magnitude = np.sqrt(
        np.mean(tail_direction.reshape(len(tail_direction), -1).astype(np.float64) ** 2, axis=1)
    ).astype(np.float32)

    risk_gate = np.clip((risk - 0.72) / 0.28, 0.0, 1.0).astype(np.float32)
    align_gate = np.clip((tail_alignment - 0.82) / 0.16, 0.0, 1.0).astype(np.float32)
    magnitude_gate = np.clip((tail_magnitude - 0.00045) / 0.00105, 0.0, 1.0).astype(
        np.float32
    )
    # v160 is nearly tied but actual output-support diagnostics show that its
    # direction often reverses the v138->v140 gain.  Treat strong anti-frontier
    # disagreement as useful evidence that the v139 tail is a different signal,
    # but do not require it everywhere.
    anti_frontier_gate = np.clip((-tail_frontier_alignment - 0.10) / 0.55, 0.0, 1.0).astype(
        np.float32
    )

    regime_scale = np.zeros(len(regimes), dtype=np.float32)
    regime_scale[regimes == "medium"] = 0.22
    regime_scale[regimes == "heavy"] = 1.00
    action = (
        regime_scale
        * risk_gate
        * align_gate
        * magnitude_gate
        * np.clip(0.70 + 0.30 * anti_frontier_gate, 0.0, 1.0)
    ).astype(np.float32)

    alpha = np.zeros(len(regimes), dtype=np.float32)
    alpha[regimes == "medium"] = 0.18 * action[regimes == "medium"]
    alpha[regimes == "heavy"] = 0.62 * action[regimes == "heavy"]

    log_target = np.log(np.maximum(anchor, FLOOR)).astype(np.float32)
    # Use the unweighted log-ratio for the actual support move; the weighted
    # direction above is only for stable sign/magnitude diagnostics.
    raw_tail_ratio = np.log(np.maximum(supports[139], FLOOR)) - np.log(
        np.maximum(anchor, FLOOR)
    )
    raw_tail_ratio = np.clip(raw_tail_ratio, -2.2, 2.2).astype(np.float32)
    target = np.exp(log_target + alpha[:, None, None, None] * raw_tail_ratio).astype(
        np.float32
    )
    target = normalize_support(target)
    target[np.isin(regimes, ["clear", "light"])] = anchor[
        np.isin(regimes, ["clear", "light"])
    ]
    np.save(TARGET_OUT, target)

    max_weight = np.clip(0.74 + 0.24 * action, 0.74, 0.98).astype(np.float32)
    weights = np.empty((len(action), 8), dtype=np.float32)
    weights[:, 0] = max_weight
    weights[:, 1:] = (1.0 - max_weight[:, None]) / 7.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    np.save(WEIGHTS_OUT, weights)

    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        risk=risk,
        risk_gate=risk_gate,
        tail_alignment=tail_alignment,
        frontier_alignment=frontier_alignment,
        tail_frontier_alignment=tail_frontier_alignment,
        tail_magnitude=tail_magnitude,
        align_gate=align_gate,
        magnitude_gate=magnitude_gate,
        anti_frontier_gate=anti_frontier_gate,
        action=action,
        alpha=alpha,
        weights=weights,
    )

    args = argparse.Namespace(
        base=str(PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy"),
        output=str(OUTPUT),
        beta=140.3,
        gamma=0.10,
        eta="clear:0.0,light:0.0,medium:0.20,heavy:0.92",
        min_gate=0.05,
        max_gate=2.70,
        conf_center=0.32,
        conf_width=0.34,
        entropy_low=0.04,
        entropy_high=0.94,
        gain_clip="0.40,3.18",
        top_gain_clip="0.50,2.50",
        top_components=5632,
        target_floor=FLOOR,
        project_batch=20,
    )
    projection_report = projection(target, weights, regimes, pos_test, args)
    comparison = compare_outputs(
        OUTPUT, PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", regimes
    )
    if comparison["shape"] != [500, 256, 4, 192] or comparison["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison['shape']} {comparison['dtype']}")
    if not comparison["finite"]:
        raise ValueError("v161 output contains non-finite values")

    target_cos = row_cosine(target, anchor)
    active_by_regime = {
        regime: {
            "count": int(np.sum(regimes == regime)),
            "active_fraction": float(np.mean(action[regimes == regime] > 1e-6)),
            "action": summarize(action[regimes == regime]),
            "alpha": summarize(alpha[regimes == regime]),
            "tail_alignment": summarize(tail_alignment[regimes == regime]),
            "frontier_alignment": summarize(frontier_alignment[regimes == regime]),
        }
        for regime in REGIMES
        if np.any(regimes == regime)
    }
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "distill the same-direction high-risk tail from rejected v139 instead "
            "of reviving v139 wholesale; actual output-support diagnostics show "
            "v139-v140 is aligned with the confirmed v138->v140 online-positive "
            "direction on heavy points, while v160 is mostly anti-aligned"
        ),
        "new_information_used": {
            "confirmed_positive_direction": "v138 online 0.500793 -> v140 online 0.501352",
            "tail_source": "v139 rejected local output support, not direct h=-3 code replay",
            "anti_example": "v160 online 0.501338 actual support direction is near-tied but mostly anti-aligned",
        },
        "excluded": {
            "v145_local_h_field": "actual output-support direction is mostly anti-aligned with v138->v140 on heavy",
            "full_v139_revival": "v139 local diagnostics showed heavy support damage; v161 gates and clips the tail",
            "phase_or_power_relearning": True,
        },
        "parameters": {
            "risk_gate": "(risk - 0.72) / 0.28 clipped",
            "align_gate": "(tail_alignment - 0.82) / 0.16 clipped",
            "magnitude_gate": "(tail_magnitude - 0.00045) / 0.00105 clipped",
            "anti_frontier_gate": "(-tail_frontier_alignment - 0.10) / 0.55 clipped",
            "alpha_medium_max": 0.18,
            "alpha_heavy_max": 0.62,
            "raw_tail_log_ratio_clip": [-2.2, 2.2],
            "projection_args": {
                key: (PATHS.relative(value) if key in {"base", "output"} else value)
                for key, value in vars(args).items()
            },
        },
        "diagnostics": {
            "risk": summarize(risk),
            "tail_alignment_to_v138_v140": summarize(tail_alignment),
            "frontier_alignment_to_v138_v140": summarize(frontier_alignment),
            "tail_frontier_alignment": summarize(tail_frontier_alignment),
            "tail_magnitude": summarize(tail_magnitude),
            "action": summarize(action),
            "alpha": summarize(alpha),
            "active_fraction": float(np.mean(action > 1e-6)),
            "active_by_regime": active_by_regime,
        },
        "target_diagnostics": {
            "target_to_v140_output_support": summarize(target_cos),
            "target_h_l1_change": summarize(l1_h_change(target, anchor)),
            "target_to_v139_output_support": summarize(row_cosine(target, supports[139])),
        },
        "projection": projection_report,
        "comparison_to_v140": comparison,
        "preserved": {
            "base_complex": "v140",
            "clear_light_points": "exact v140 output support and eta=0",
            "phase": "inherited from v140 through positive AD amplitude projection",
            "total_power": "projection renormalizes each point to v140 AD total power",
            "pol_ue_allocation": "unchanged except projection-induced positive amplitude scaling",
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
            (__file__, "generate_v161_hminus3_tail_distillation.py"),
            (PATHS.code / "generate_v160_frontier_consensus_overdrive.py", "generate_v160_frontier_consensus_overdrive.py"),
            (PATHS.code / "generate_v159_frontier_consensus_support_soup.py", "generate_v159_frontier_consensus_support_soup.py"),
            (PATHS.code / "generate_v140_balanced_hminus2_overdrive.py", "generate_v140_balanced_hminus2_overdrive.py"),
            (PATHS.code / "generate_v139_ultra_geodesic_multishift.py", "generate_v139_ultra_geodesic_multishift.py"),
            (PATHS.code / "generate_v138_extreme_geodesic_shift_overdrive.py", "generate_v138_extreme_geodesic_shift_overdrive.py"),
            (PATHS.code / "generate_v98_path_ensemble_graph_attention_joint_ad.py", "generate_v98_path_ensemble_graph_attention_joint_ad.py"),
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
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
