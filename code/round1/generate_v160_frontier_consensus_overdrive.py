#!/usr/bin/env python3
"""Generate v160: aggressive online-frontier consensus overdrive.

v159 proved that a conservative consensus soup is too weak: its output NMSE to
v140 was only 1.7e-8 because projection confidence collapsed.  v160 keeps the
same single hypothesis but removes the over-strict divergence/confidence gate.

Hypothesis:
    v143/v150/v152 are each slightly below v140, but they are nearly tied with
    it online and encode different weak corrections.  Their geometric AD
    support consensus may be useful if applied only to medium/heavy points
    with a deliberately strong support-only projection.

Preserved:
    v140 complex phase, point total power, clear/light points, and pol/UE
    allocation.
"""

from __future__ import annotations

import argparse
import json
import zipfile

import numpy as np

from generate_v87_easy_field_projection import row_cos, summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from generate_v159_frontier_consensus_support_soup import (
    CONSENSUS_OUT as V159_CONSENSUS_OUT,
    FLOOR,
    FRONTIER,
    REGIMES,
    collapsed_support_from_output,
    compare_outputs,
    digest,
    geometric_mix,
    normalize_support,
)
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v160_frontier_consensus_overdrive"
OUT_DIR = PATHS.cache_dir(160)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(160)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v160_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v160_frontier_consensus_overdrive_target.npy"
CONSENSUS_OUT = OUT_DIR / "v160_frontier_consensus_alt_support.npy"
WEIGHTS_OUT = OUT_DIR / "v160_frontier_consensus_overdrive_weights.npy"
ACTION_OUT = OUT_DIR / "v160_frontier_consensus_overdrive_actions.npz"
REPORT = REPORT_DIR / "v160_frontier_consensus_overdrive_generate.json"
ZIP_OUT = PATHS.round1_versions / "v160_frontier_consensus_overdrive_candidate_pending.zip"


def support_cache_for(name: str) -> str:
    # Reuse v159 support caches when present; they are deterministic collapsed
    # AD supports computed from immutable version outputs.
    old = PATHS.cache_dir(159) / f"{name}_collapsed_support.npy"
    return str(old if old.is_file() else OUT_DIR / f"{name}_collapsed_support.npy")


def main() -> None:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked_test = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "test_blocked"
    ].astype(np.float32)
    regimes = regime_of(blocked_test)
    risk = np.load(PATHS.cache_dir(140) / "v140_geodesic_risk.npy").astype(np.float32)

    supports: dict[str, np.ndarray] = {}
    for name, item in FRONTIER.items():
        supports[name] = collapsed_support_from_output(
            item["path"], pos_test, PATHS.resolve(support_cache_for(name))
        )
    anchor = supports["v140"]

    alt_names = ["v143", "v150", "v152"]
    alt_weights = np.asarray([0.28, 0.22, 0.50], dtype=np.float32)
    if V159_CONSENSUS_OUT.is_file():
        alt_consensus = normalize_support(np.load(V159_CONSENSUS_OUT).astype(np.float32))
        print(f"reuse alt consensus {PATHS.relative(V159_CONSENSUS_OUT)}", flush=True)
    else:
        alt_consensus = geometric_mix([supports[name] for name in alt_names], alt_weights)
    np.save(CONSENSUS_OUT, alt_consensus)

    cos_alt_to_anchor = {
        name: row_cos(supports[name], anchor).astype(np.float32) for name in alt_names
    }
    cos_alt_to_consensus = {
        name: row_cos(supports[name], alt_consensus).astype(np.float32) for name in alt_names
    }
    agreement = np.mean(
        np.stack([cos_alt_to_consensus[name] for name in alt_names], axis=1), axis=1
    ).astype(np.float32)
    consensus_to_anchor = row_cos(alt_consensus, anchor).astype(np.float32)
    divergence = np.clip(1.0 - consensus_to_anchor, 0.0, 1.0).astype(np.float32)

    # v159 was too conservative.  v160 treats the near-frontier agreement as
    # enough evidence and uses risk mainly to decide strength, not eligibility.
    agreement_gate = np.clip((agreement - 0.982) / 0.015, 0.0, 1.0).astype(np.float32)
    risk_gate = np.clip((risk - 0.16) / 0.72, 0.0, 1.0).astype(np.float32)
    divergence_boost = np.clip((divergence - 0.0004) / 0.018, 0.0, 1.0).astype(np.float32)
    regime_scale = np.zeros(len(regimes), dtype=np.float32)
    regime_scale[regimes == "medium"] = 0.62
    regime_scale[regimes == "heavy"] = 1.00
    action = agreement_gate * regime_scale * np.clip(
        0.30 + 0.55 * risk_gate + 0.15 * divergence_boost, 0.0, 1.0
    )
    action = np.clip(action, 0.0, 1.0).astype(np.float32)

    alpha = np.zeros(len(regimes), dtype=np.float32)
    alpha[regimes == "medium"] = 0.36 * action[regimes == "medium"]
    alpha[regimes == "heavy"] = 0.62 * action[regimes == "heavy"]

    target = np.exp(
        (1.0 - alpha[:, None, None, None]) * np.log(np.maximum(anchor, FLOOR))
        + alpha[:, None, None, None] * np.log(np.maximum(alt_consensus, FLOOR))
    ).astype(np.float32)
    target = normalize_support(target)
    target[np.isin(regimes, ["clear", "light"])] = anchor[np.isin(regimes, ["clear", "light"])]
    np.save(TARGET_OUT, target)

    max_weight = np.clip(0.72 + 0.25 * action, 0.72, 0.97).astype(np.float32)
    weights = np.empty((len(action), 8), dtype=np.float32)
    weights[:, 0] = max_weight
    weights[:, 1:] = (1.0 - max_weight[:, None]) / 7.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    np.save(WEIGHTS_OUT, weights)

    np.savez_compressed(
        ACTION_OUT,
        risk=risk,
        regimes=regimes.astype("U8"),
        agreement=agreement,
        consensus_to_anchor=consensus_to_anchor,
        divergence=divergence,
        agreement_gate=agreement_gate,
        risk_gate=risk_gate,
        divergence_boost=divergence_boost,
        action=action,
        alpha=alpha,
        weights=weights,
        **{f"cos_{name}_to_v140": cos_alt_to_anchor[name] for name in alt_names},
        **{f"cos_{name}_to_consensus": cos_alt_to_consensus[name] for name in alt_names},
    )

    args = argparse.Namespace(
        base=str(FRONTIER["v140"]["path"]),
        output=str(OUTPUT),
        beta=140.3,
        gamma=0.10,
        eta="clear:0.0,light:0.0,medium:0.46,heavy:0.84",
        min_gate=0.05,
        max_gate=2.45,
        conf_center=0.34,
        conf_width=0.34,
        entropy_low=0.04,
        entropy_high=0.94,
        gain_clip="0.40,3.05",
        top_gain_clip="0.50,2.38",
        top_components=5632,
        target_floor=FLOOR,
        project_batch=20,
    )
    projection_report = projection(target, weights, regimes, pos_test, args)
    comparison = compare_outputs(OUTPUT, FRONTIER["v140"]["path"], regimes)
    if comparison["shape"] != [500, 256, 4, 192] or comparison["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison['shape']} {comparison['dtype']}")
    if not comparison["finite"]:
        raise ValueError("v160 output contains non-finite values")

    target_cos = row_cos(target, anchor)
    action_by_regime = {
        regime: {
            "count": int(np.sum(regimes == regime)),
            "active_fraction": float(np.mean(action[regimes == regime] > 1e-6)),
            "alpha": summarize(alpha[regimes == regime]),
            "action": summarize(action[regimes == regime]),
        }
        for regime in REGIMES
        if np.any(regimes == regime)
    }
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "a deliberately stronger support-only projection toward the shared "
            "v143/v150/v152 online-frontier consensus can combine their weak "
            "complementary signals without inheriting their full individual errors"
        ),
        "why_after_v159": (
            "v159 was rejected locally because output NMSE to v140 was only "
            "1.7e-8; v160 raises action, alpha, confidence weights and eta"
        ),
        "new_information_used": {
            name: {
                "online_score": item["score"],
                "role": item["role"],
                "output": PATHS.relative(item["path"]),
            }
            for name, item in FRONTIER.items()
        },
        "excluded": {
            "v155_metric_projection": "online farther from v140 and changes clear/light",
            "v158_direct_path_peak_prior": "online failed by over-pushing heavy support",
            "complex_phase_regression": True,
            "total_power_recalibration": True,
        },
        "parameters": {
            "alt_names": alt_names,
            "alt_weights": alt_weights.tolist(),
            "alpha_medium_max": 0.36,
            "alpha_heavy_max": 0.62,
            "eta": args.eta,
            "gain_clip": args.gain_clip,
            "top_gain_clip": args.top_gain_clip,
            "top_components": args.top_components,
        },
        "consensus_diagnostics": {
            "agreement": summarize(agreement),
            "consensus_to_v140": summarize(consensus_to_anchor),
            "divergence": summarize(divergence),
            "risk": summarize(risk),
            "action": summarize(action),
            "alpha": summarize(alpha),
            "active_fraction": float(np.mean(action > 1e-6)),
            "active_by_regime": action_by_regime,
            "cos_alt_to_v140": {
                name: summarize(cos_alt_to_anchor[name]) for name in alt_names
            },
            "cos_alt_to_consensus": {
                name: summarize(cos_alt_to_consensus[name]) for name in alt_names
            },
        },
        "target_diagnostics": {
            "target_to_v140": summarize(target_cos),
            "target_to_alt_consensus": summarize(row_cos(target, alt_consensus)),
            "target_h_l1_change": summarize(
                np.sum(
                    np.abs(
                        target.sum(axis=-1).reshape(len(target), -1)
                        - anchor.sum(axis=-1).reshape(len(anchor), -1)
                    ),
                    axis=1,
                )
            ),
        },
        "projection": projection_report,
        "comparison_to_v140": comparison,
        "preserved": {
            "base_complex": "v140",
            "clear_light_points": "exact v140 target and eta=0",
            "phase": "inherited from v140 through positive AD amplitude projection",
            "total_power": "projection renormalizes each point to v140 AD total power",
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET_OUT),
            "alt_consensus": PATHS.relative(CONSENSUS_OUT),
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
            (__file__, "generate_v160_frontier_consensus_overdrive.py"),
            (PATHS.code / "generate_v159_frontier_consensus_support_soup.py", "generate_v159_frontier_consensus_support_soup.py"),
            (PATHS.code / "generate_v152_reverse_source_local_h_field.py", "generate_v152_reverse_source_local_h_field.py"),
            (PATHS.code / "generate_v150_los_safe_continuous_action_controller.py", "generate_v150_los_safe_continuous_action_controller.py"),
            (PATHS.code / "generate_v143_transductive_geodesic_risk.py", "generate_v143_transductive_geodesic_risk.py"),
            (PATHS.code / "generate_v140_balanced_hminus2_overdrive.py", "generate_v140_balanced_hminus2_overdrive.py"),
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
