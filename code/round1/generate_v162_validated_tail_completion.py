#!/usr/bin/env python3
"""Generate v162: second-stage completion of the online-validated v139 tail.

Hypothesis:
    v161 made v139's high-risk heavy tail online-positive by filtering it with
    actual-output AD-support direction.  Now v140->v161 is itself an
    online-validated direction.  v162 uses that new label to take one more
    bounded step from v161 toward the remaining v139 tail, but only where:

    * the point was already moved by v161;
    * the remaining v161->v139 tail is aligned with v140->v161;
    * the point is high-risk heavy;
    * the residual direction is not just the failed v160/v145 direction.

This is intentionally aggressive, but it is not a blind h=-3 replay.
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
from generate_v161_hminus3_tail_distillation import (
    l1_h_change,
    row_cosine,
    weighted_log_direction,
)
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v162_validated_tail_completion"
OUT_DIR = PATHS.cache_dir(162)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(162)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v162_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v162_validated_tail_completion_target.npy"
WEIGHTS_OUT = OUT_DIR / "v162_validated_tail_completion_weights.npy"
ACTION_OUT = OUT_DIR / "v162_validated_tail_completion_actions.npz"
REPORT = REPORT_DIR / "v162_validated_tail_completion_generate.json"
ZIP_OUT = PATHS.round1_versions / "v162_validated_tail_completion_candidate_pending.zip"


def support_cache(version: int) -> str:
    probe = OUT_DIR / f"probe_v{version}_collapsed_support.npy"
    if probe.is_file():
        return str(probe)
    return str(OUT_DIR / f"v162_v{version}_collapsed_support.npy")


def main() -> None:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked_test = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "test_blocked"
    ].astype(np.float32)
    regimes = regime_of(blocked_test)
    risk = np.load(PATHS.cache_dir(140) / "v140_geodesic_risk.npy").astype(np.float32)

    support_paths = {
        139: PATHS.cache_dir(139) / "v139_Round1_Test_Channel.npy",
        140: PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy",
        145: PATHS.cache_dir(145) / "v145_Round1_Test_Channel.npy",
        160: PATHS.cache_dir(160) / "v160_Round1_Test_Channel.npy",
        161: PATHS.cache_dir(161) / "v161_Round1_Test_Channel.npy",
    }
    supports = {
        version: collapsed_support_from_output(
            path, pos_test, PATHS.resolve(support_cache(version))
        )
        for version, path in support_paths.items()
    }
    anchor = supports[161]

    v161_actions = np.load(PATHS.cache_dir(161) / "v161_hminus3_tail_distillation_actions.npz")
    previous_alpha = v161_actions["alpha"].astype(np.float32)
    previous_action = v161_actions["action"].astype(np.float32)

    validated_direction = weighted_log_direction(supports[140], supports[161])
    remaining_tail = weighted_log_direction(supports[161], supports[139])
    frontier_tail = weighted_log_direction(supports[161], supports[160])
    local_h_tail = weighted_log_direction(supports[161], supports[145])

    remaining_alignment = row_cosine(remaining_tail, validated_direction)
    frontier_alignment = row_cosine(remaining_tail, frontier_tail)
    local_h_alignment = row_cosine(remaining_tail, local_h_tail)
    remaining_magnitude = np.sqrt(
        np.mean(
            remaining_tail.reshape(len(remaining_tail), -1).astype(np.float64) ** 2,
            axis=1,
        )
    ).astype(np.float32)

    risk_gate = np.clip((risk - 0.74) / 0.24, 0.0, 1.0).astype(np.float32)
    align_gate = np.clip((remaining_alignment - 0.78) / 0.20, 0.0, 1.0).astype(
        np.float32
    )
    magnitude_gate = np.clip((remaining_magnitude - 0.00008) / 0.00042, 0.0, 1.0).astype(
        np.float32
    )
    prior_gate = np.clip((previous_alpha - 0.03) / 0.30, 0.0, 1.0).astype(np.float32)
    anti_frontier_gate = np.clip((-frontier_alignment - 0.12) / 0.55, 0.0, 1.0).astype(
        np.float32
    )
    anti_local_gate = np.clip((-local_h_alignment - 0.08) / 0.55, 0.0, 1.0).astype(
        np.float32
    )

    regime_scale = np.zeros(len(regimes), dtype=np.float32)
    regime_scale[regimes == "heavy"] = 1.0
    action = (
        regime_scale
        * risk_gate
        * align_gate
        * magnitude_gate
        * prior_gate
        * np.clip(0.64 + 0.20 * anti_frontier_gate + 0.16 * anti_local_gate, 0.0, 1.0)
    ).astype(np.float32)

    # This deliberately allows near-completion of the validated residual tail.
    # Overshoot is small and only happens at the highest-confidence heavy tail.
    alpha = np.zeros(len(regimes), dtype=np.float32)
    alpha[regimes == "heavy"] = 1.65 * action[regimes == "heavy"]

    raw_tail_ratio = np.log(np.maximum(supports[139], FLOOR)) - np.log(
        np.maximum(anchor, FLOOR)
    )
    raw_tail_ratio = np.clip(raw_tail_ratio, -2.20, 2.20).astype(np.float32)
    target = np.exp(
        np.log(np.maximum(anchor, FLOOR)).astype(np.float32)
        + alpha[:, None, None, None] * raw_tail_ratio
    ).astype(np.float32)
    target = normalize_support(target)
    target[regimes != "heavy"] = anchor[regimes != "heavy"]
    np.save(TARGET_OUT, target)

    max_weight = np.clip(0.78 + 0.20 * action, 0.78, 0.98).astype(np.float32)
    weights = np.empty((len(action), 8), dtype=np.float32)
    weights[:, 0] = max_weight
    weights[:, 1:] = (1.0 - max_weight[:, None]) / 7.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    np.save(WEIGHTS_OUT, weights)

    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        risk=risk,
        previous_action=previous_action,
        previous_alpha=previous_alpha,
        remaining_alignment=remaining_alignment,
        frontier_alignment=frontier_alignment,
        local_h_alignment=local_h_alignment,
        remaining_magnitude=remaining_magnitude,
        risk_gate=risk_gate,
        align_gate=align_gate,
        magnitude_gate=magnitude_gate,
        prior_gate=prior_gate,
        anti_frontier_gate=anti_frontier_gate,
        anti_local_gate=anti_local_gate,
        action=action,
        alpha=alpha,
        weights=weights,
    )

    args = argparse.Namespace(
        base=str(PATHS.cache_dir(161) / "v161_Round1_Test_Channel.npy"),
        output=str(OUTPUT),
        beta=140.3,
        gamma=0.10,
        eta="clear:0.0,light:0.0,medium:0.0,heavy:1.08",
        min_gate=0.05,
        max_gate=2.88,
        conf_center=0.32,
        conf_width=0.32,
        entropy_low=0.04,
        entropy_high=0.94,
        gain_clip="0.38,3.20",
        top_gain_clip="0.50,2.55",
        top_components=5632,
        target_floor=FLOOR,
        project_batch=20,
    )
    projection_report = projection(target, weights, regimes, pos_test, args)
    comparison_to_v161 = compare_outputs(
        OUTPUT, PATHS.cache_dir(161) / "v161_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v140 = compare_outputs(
        OUTPUT, PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", regimes
    )
    if comparison_to_v161["shape"] != [500, 256, 4, 192] or comparison_to_v161["dtype"] != "complex64":
        raise ValueError(
            f"invalid output layout: {comparison_to_v161['shape']} {comparison_to_v161['dtype']}"
        )
    if not comparison_to_v161["finite"]:
        raise ValueError("v162 output contains non-finite values")

    active_by_regime = {
        regime: {
            "count": int(np.sum(regimes == regime)),
            "active_fraction": float(np.mean(action[regimes == regime] > 1e-6)),
            "action": summarize(action[regimes == regime]),
            "alpha": summarize(alpha[regimes == regime]),
            "remaining_alignment": summarize(remaining_alignment[regimes == regime]),
            "remaining_magnitude": summarize(remaining_magnitude[regimes == regime]),
        }
        for regime in REGIMES
        if np.any(regimes == regime)
    }
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v161 online feedback turns v140->v161 into a validated positive "
            "support direction; v162 completes the remaining v161->v139 heavy "
            "tail only where that residual is still aligned with the validated "
            "direction and already had v161 support"
        ),
        "new_information_used": {
            "online_label": "v161 online_0.502273 > v140 online_0.501352",
            "validated_direction": "actual output AD support v140 -> v161",
            "remaining_tail": "actual output AD support v161 -> v139",
            "negative_direction_controls": "v161 -> v160 and v161 -> v145 remain mostly anti-aligned",
        },
        "excluded": {
            "full_v139_replay": True,
            "new_local_h_residual": True,
            "phase_or_power_relearning": True,
            "medium_or_clear_light_changes": True,
        },
        "parameters": {
            "risk_gate": "(risk - 0.74) / 0.24 clipped",
            "align_gate": "(remaining_alignment - 0.78) / 0.20 clipped",
            "magnitude_gate": "(remaining_magnitude - 0.00008) / 0.00042 clipped",
            "prior_gate": "(v161_alpha - 0.03) / 0.30 clipped",
            "anti_frontier_gate": "(-frontier_alignment - 0.12) / 0.55 clipped",
            "anti_local_gate": "(-local_h_alignment - 0.08) / 0.55 clipped",
            "alpha_heavy_max": 1.65,
            "raw_tail_log_ratio_clip": [-2.20, 2.20],
            "projection_args": {
                key: (PATHS.relative(value) if key in {"base", "output"} else value)
                for key, value in vars(args).items()
            },
        },
        "diagnostics": {
            "risk": summarize(risk),
            "previous_alpha": summarize(previous_alpha),
            "remaining_alignment_to_v140_v161": summarize(remaining_alignment),
            "frontier_alignment": summarize(frontier_alignment),
            "local_h_alignment": summarize(local_h_alignment),
            "remaining_magnitude": summarize(remaining_magnitude),
            "action": summarize(action),
            "alpha": summarize(alpha),
            "active_fraction": float(np.mean(action > 1e-6)),
            "active_by_regime": active_by_regime,
        },
        "target_diagnostics": {
            "target_to_v161_output_support": summarize(row_cosine(target, anchor)),
            "target_to_v139_output_support": summarize(row_cosine(target, supports[139])),
            "target_h_l1_change_to_v161": summarize(l1_h_change(target, anchor)),
        },
        "projection": projection_report,
        "comparison_to_v161": comparison_to_v161,
        "comparison_to_v140": comparison_to_v140,
        "preserved": {
            "base_complex": "v161",
            "non_heavy_points": "exact v161 target and eta=0",
            "phase": "inherited from v161 through positive AD amplitude projection",
            "total_power": "projection renormalizes each point to v161 AD total power",
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
            (__file__, "generate_v162_validated_tail_completion.py"),
            (PATHS.code / "generate_v161_hminus3_tail_distillation.py", "generate_v161_hminus3_tail_distillation.py"),
            (PATHS.code / "generate_v160_frontier_consensus_overdrive.py", "generate_v160_frontier_consensus_overdrive.py"),
            (PATHS.code / "generate_v140_balanced_hminus2_overdrive.py", "generate_v140_balanced_hminus2_overdrive.py"),
            (PATHS.code / "generate_v139_ultra_geodesic_multishift.py", "generate_v139_ultra_geodesic_multishift.py"),
            (PATHS.code / "generate_v98_path_ensemble_graph_attention_joint_ad.py", "generate_v98_path_ensemble_graph_attention_joint_ad.py"),
            (PATHS.code / "generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
            (PATHS.code / "generate_v159_frontier_consensus_support_soup.py", "generate_v159_frontier_consensus_support_soup.py"),
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
