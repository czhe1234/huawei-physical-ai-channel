#!/usr/bin/env python3
"""Generate v163: online-validated momentum overdrive from v162.

Hypothesis:
    v162 online feedback confirms that v161->v162 is itself an online-positive
    support direction.  The remaining v162->v139 direction is now mostly
    anti-aligned on heavy points, so v163 must not keep chasing v139.  Instead,
    it extrapolates the validated v161->v162 momentum on high-risk heavy points
    where that momentum:

    * is chained with the previous positive v140->v161 direction;
    * was already active in v162;
    * points away from the failed v160/v145 directions;
    * is not simply an unfiltered replay of the original v139 tail.

Preserved:
    v162 complex phase, point total power, non-heavy points, and pol/UE
    allocation.  Only selected heavy collapsed joint AD support is changed.
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


VERSION = "v163_validated_momentum_overdrive"
OUT_DIR = PATHS.cache_dir(163)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(163)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v163_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v163_validated_momentum_overdrive_target.npy"
WEIGHTS_OUT = OUT_DIR / "v163_validated_momentum_overdrive_weights.npy"
ACTION_OUT = OUT_DIR / "v163_validated_momentum_overdrive_actions.npz"
REPORT = REPORT_DIR / "v163_validated_momentum_overdrive_generate.json"
ZIP_OUT = PATHS.round1_versions / "v163_validated_momentum_overdrive_candidate_pending.zip"


def support_cache(version: int) -> str:
    probe = OUT_DIR / f"probe_v{version}_collapsed_support.npy"
    if probe.is_file():
        return str(probe)
    return str(OUT_DIR / f"v163_v{version}_collapsed_support.npy")


def rms_direction(direction: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.mean(direction.reshape(len(direction), -1).astype(np.float64) ** 2, axis=1)
    ).astype(np.float32)


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
        162: PATHS.cache_dir(162) / "v162_Round1_Test_Channel.npy",
    }
    supports = {
        version: collapsed_support_from_output(
            path, pos_test, PATHS.resolve(support_cache(version))
        )
        for version, path in support_paths.items()
    }
    anchor = supports[162]

    v162_actions = np.load(
        PATHS.cache_dir(162) / "v162_validated_tail_completion_actions.npz"
    )
    previous_action = v162_actions["action"].astype(np.float32)
    previous_alpha = v162_actions["alpha"].astype(np.float32)

    first_validated = weighted_log_direction(supports[140], supports[161])
    momentum = weighted_log_direction(supports[161], supports[162])
    to_frontier = weighted_log_direction(supports[162], supports[160])
    to_local_h = weighted_log_direction(supports[162], supports[145])
    to_v139 = weighted_log_direction(supports[162], supports[139])

    chain_alignment = row_cosine(momentum, first_validated)
    frontier_alignment = row_cosine(momentum, to_frontier)
    local_h_alignment = row_cosine(momentum, to_local_h)
    v139_alignment = row_cosine(momentum, to_v139)
    momentum_magnitude = rms_direction(momentum)

    regime_scale = np.zeros(len(regimes), dtype=np.float32)
    regime_scale[regimes == "heavy"] = 1.0

    risk_gate = np.clip((risk - 0.76) / 0.22, 0.0, 1.0).astype(np.float32)
    prior_gate = np.clip((previous_action - 0.10) / 0.72, 0.0, 1.0).astype(
        np.float32
    )
    chain_gate = np.clip((chain_alignment - 0.28) / 0.62, 0.0, 1.0).astype(
        np.float32
    )
    magnitude_gate = np.clip((momentum_magnitude - 0.00008) / 0.00055, 0.0, 1.0).astype(
        np.float32
    )
    anti_frontier_gate = np.clip((-frontier_alignment - 0.12) / 0.56, 0.0, 1.0).astype(
        np.float32
    )
    anti_local_gate = np.clip((-local_h_alignment - 0.08) / 0.56, 0.0, 1.0).astype(
        np.float32
    )
    # After v162, v139 is no longer the target.  Prefer points where v139 is
    # behind or sideways relative to the online-validated momentum.
    v139_boundary_gate = np.clip((0.32 - v139_alignment) / 0.90, 0.0, 1.0).astype(
        np.float32
    )

    action = (
        regime_scale
        * risk_gate
        * prior_gate
        * chain_gate
        * magnitude_gate
        * np.clip(0.56 + 0.24 * anti_frontier_gate + 0.20 * anti_local_gate, 0.0, 1.0)
        * np.clip(0.72 + 0.28 * v139_boundary_gate, 0.0, 1.0)
    ).astype(np.float32)

    extra_alpha = np.zeros(len(regimes), dtype=np.float32)
    extra_alpha[regimes == "heavy"] = 0.78 * action[regimes == "heavy"]

    raw_momentum_ratio = np.log(np.maximum(supports[162], FLOOR)) - np.log(
        np.maximum(supports[161], FLOOR)
    )
    raw_momentum_ratio = np.clip(raw_momentum_ratio, -1.80, 1.80).astype(np.float32)
    target = np.exp(
        np.log(np.maximum(anchor, FLOOR)).astype(np.float32)
        + extra_alpha[:, None, None, None] * raw_momentum_ratio
    ).astype(np.float32)
    target = normalize_support(target)
    target[regimes != "heavy"] = anchor[regimes != "heavy"]
    np.save(TARGET_OUT, target)

    max_weight = np.clip(0.80 + 0.18 * action, 0.80, 0.98).astype(np.float32)
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
        chain_alignment=chain_alignment,
        frontier_alignment=frontier_alignment,
        local_h_alignment=local_h_alignment,
        v139_alignment=v139_alignment,
        momentum_magnitude=momentum_magnitude,
        risk_gate=risk_gate,
        prior_gate=prior_gate,
        chain_gate=chain_gate,
        magnitude_gate=magnitude_gate,
        anti_frontier_gate=anti_frontier_gate,
        anti_local_gate=anti_local_gate,
        v139_boundary_gate=v139_boundary_gate,
        action=action,
        extra_alpha=extra_alpha,
        weights=weights,
    )

    args = argparse.Namespace(
        base=str(PATHS.cache_dir(162) / "v162_Round1_Test_Channel.npy"),
        output=str(OUTPUT),
        beta=140.3,
        gamma=0.10,
        eta="clear:0.0,light:0.0,medium:0.0,heavy:0.96",
        min_gate=0.05,
        max_gate=2.76,
        conf_center=0.32,
        conf_width=0.32,
        entropy_low=0.04,
        entropy_high=0.94,
        gain_clip="0.40,3.12",
        top_gain_clip="0.52,2.42",
        top_components=5632,
        target_floor=FLOOR,
        project_batch=20,
    )
    projection_report = projection(target, weights, regimes, pos_test, args)
    comparison_to_v162 = compare_outputs(
        OUTPUT, PATHS.cache_dir(162) / "v162_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v161 = compare_outputs(
        OUTPUT, PATHS.cache_dir(161) / "v161_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v140 = compare_outputs(
        OUTPUT, PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", regimes
    )
    if comparison_to_v162["shape"] != [500, 256, 4, 192] or comparison_to_v162["dtype"] != "complex64":
        raise ValueError(
            f"invalid output layout: {comparison_to_v162['shape']} {comparison_to_v162['dtype']}"
        )
    if not comparison_to_v162["finite"]:
        raise ValueError("v163 output contains non-finite values")

    active_by_regime = {
        regime: {
            "count": int(np.sum(regimes == regime)),
            "active_fraction": float(np.mean(action[regimes == regime] > 1e-6)),
            "action": summarize(action[regimes == regime]),
            "extra_alpha": summarize(extra_alpha[regimes == regime]),
            "chain_alignment": summarize(chain_alignment[regimes == regime]),
            "frontier_alignment": summarize(frontier_alignment[regimes == regime]),
            "local_h_alignment": summarize(local_h_alignment[regimes == regime]),
            "v139_alignment": summarize(v139_alignment[regimes == regime]),
            "momentum_magnitude": summarize(momentum_magnitude[regimes == regime]),
        }
        for regime in REGIMES
        if np.any(regimes == regime)
    }

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v162 online feedback makes v161->v162 a newly validated positive "
            "support momentum; v163 extrapolates that momentum on selected "
            "high-risk heavy points instead of continuing to chase v139"
        ),
        "new_information_used": {
            "online_label": "v162 online_0.503192 > v161 online_0.502273",
            "validated_momentum": "actual output AD support v161 -> v162",
            "saturation_signal": "remaining v162 -> v139 is mostly anti-aligned on heavy points",
            "negative_direction_controls": "v162 -> v160 and v162 -> v145 are mostly opposite to the validated momentum",
        },
        "excluded": {
            "full_v139_replay": True,
            "v162_to_v139_completion": True,
            "new_local_h_residual": True,
            "phase_or_power_relearning": True,
            "medium_or_clear_light_changes": True,
        },
        "parameters": {
            "risk_gate": "(risk - 0.76) / 0.22 clipped",
            "prior_gate": "(v162_action - 0.10) / 0.72 clipped",
            "chain_gate": "(chain_alignment - 0.28) / 0.62 clipped",
            "magnitude_gate": "(momentum_magnitude - 0.00008) / 0.00055 clipped",
            "anti_frontier_gate": "(-frontier_alignment - 0.12) / 0.56 clipped",
            "anti_local_gate": "(-local_h_alignment - 0.08) / 0.56 clipped",
            "v139_boundary_gate": "(0.32 - v139_alignment) / 0.90 clipped",
            "extra_alpha_heavy_max": 0.78,
            "raw_momentum_log_ratio_clip": [-1.80, 1.80],
            "projection_args": {
                key: (PATHS.relative(value) if key in {"base", "output"} else value)
                for key, value in vars(args).items()
            },
        },
        "diagnostics": {
            "risk": summarize(risk),
            "previous_action": summarize(previous_action),
            "previous_alpha": summarize(previous_alpha),
            "chain_alignment": summarize(chain_alignment),
            "frontier_alignment": summarize(frontier_alignment),
            "local_h_alignment": summarize(local_h_alignment),
            "v139_alignment": summarize(v139_alignment),
            "momentum_magnitude": summarize(momentum_magnitude),
            "action": summarize(action),
            "extra_alpha": summarize(extra_alpha),
            "active_fraction": float(np.mean(action > 1e-6)),
            "active_by_regime": active_by_regime,
        },
        "target_diagnostics": {
            "target_to_v162_output_support": summarize(row_cosine(target, anchor)),
            "target_to_v139_output_support": summarize(row_cosine(target, supports[139])),
            "target_h_l1_change_to_v162": summarize(l1_h_change(target, anchor)),
            "target_h_l1_change_to_v161": summarize(l1_h_change(target, supports[161])),
        },
        "projection": projection_report,
        "comparison_to_v162": comparison_to_v162,
        "comparison_to_v161": comparison_to_v161,
        "comparison_to_v140": comparison_to_v140,
        "preserved": {
            "base_complex": "v162",
            "non_heavy_points": "exact v162 target and eta=0",
            "phase": "inherited from v162 through positive AD amplitude projection",
            "total_power": "projection renormalizes each point to v162 AD total power",
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
            (__file__, "generate_v163_validated_momentum_overdrive.py"),
            (PATHS.code / "generate_v162_validated_tail_completion.py", "generate_v162_validated_tail_completion.py"),
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
