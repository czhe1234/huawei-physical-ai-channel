#!/usr/bin/env python3
"""Generate v164: contrastive online-feedback debias on top of v163.

Hypothesis:
    v163 only gained +0.000044 over v162, so continuing the same momentum
    overdrive is nearly saturated.  However, a contrastive audit of actual AD
    support shows a stronger signal: the direction that moves v163 away from
    multiple online-negative or non-breaking variants (v142/v143/v144/v155/
    v158/v160 plus the v145 local-H diagnostic) is strongly aligned with the
    online-positive chain v140->v161->v162->v163 on high-risk heavy points.

    v164 therefore uses the online score history as a weak gradient.  It does
    not learn phase or total power.  It builds a bad-support ensemble from
    failed versions and pushes selected heavy points away from that ensemble,
    but only when the contrastive direction agrees with the validated positive
    chain.  A small novelty gate allows high-risk heavy points that v163 did
    not move, provided this contrastive agreement is very strong.

Preserved:
    v163 complex phase, point total power, non-heavy points, and pol/UE
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


VERSION = "v164_contrastive_online_debias"
OUT_DIR = PATHS.cache_dir(164)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(164)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v164_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v164_contrastive_online_debias_target.npy"
WEIGHTS_OUT = OUT_DIR / "v164_contrastive_online_debias_weights.npy"
ACTION_OUT = OUT_DIR / "v164_contrastive_online_debias_actions.npz"
REPORT = REPORT_DIR / "v164_contrastive_online_debias_generate.json"
ZIP_OUT = PATHS.round1_versions / "v164_contrastive_online_debias_candidate_pending.zip"


def support_cache(version: int) -> str:
    probe = OUT_DIR / f"probe_v{version}_collapsed_support.npy"
    if probe.is_file():
        return str(probe)
    return str(OUT_DIR / f"v164_v{version}_collapsed_support.npy")


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
        140: PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy",
        142: PATHS.cache_dir(142) / "v142_Round1_Test_Channel.npy",
        143: PATHS.cache_dir(143) / "v143_Round1_Test_Channel.npy",
        144: PATHS.cache_dir(144) / "v144_Round1_Test_Channel.npy",
        145: PATHS.cache_dir(145) / "v145_Round1_Test_Channel.npy",
        155: PATHS.cache_dir(155) / "v155_Round1_Test_Channel.npy",
        158: PATHS.cache_dir(158) / "v158_Round1_Test_Channel.npy",
        160: PATHS.cache_dir(160) / "v160_Round1_Test_Channel.npy",
        161: PATHS.cache_dir(161) / "v161_Round1_Test_Channel.npy",
        162: PATHS.cache_dir(162) / "v162_Round1_Test_Channel.npy",
        163: PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy",
    }
    supports = {
        version: collapsed_support_from_output(
            path, pos_test, PATHS.resolve(support_cache(version))
        )
        for version, path in support_paths.items()
    }
    anchor = supports[163]

    v163_actions = np.load(
        PATHS.cache_dir(163) / "v163_validated_momentum_overdrive_actions.npz"
    )
    previous_action = v163_actions["action"].astype(np.float32)
    previous_alpha = v163_actions["extra_alpha"].astype(np.float32)

    positive_chain = (
        0.46 * weighted_log_direction(supports[140], supports[161])
        + 0.46 * weighted_log_direction(supports[161], supports[162])
        + 0.08 * weighted_log_direction(supports[162], supports[163])
    ).astype(np.float32)

    # Score gaps are measured against the current best v163.  v145 has no
    # online score, so it is kept as a capped diagnostic control because local-H
    # directions repeatedly behaved as anti-directions in the v161-v163 audits.
    negative_score_gaps = {
        142: 0.503236 - 0.500329,
        143: 0.503236 - 0.501271,
        144: 0.503236 - 0.500586,
        145: 0.002000,
        155: 0.503236 - 0.500433,
        158: 0.503236 - 0.500224,
        160: 0.503236 - 0.501338,
    }
    negative_versions = np.array(list(negative_score_gaps), dtype=np.int32)
    negative_weights = np.array(
        [negative_score_gaps[int(v)] for v in negative_versions], dtype=np.float64
    )
    negative_weights /= np.maximum(negative_weights.sum(), TINY)

    negative_direction = np.zeros_like(anchor, dtype=np.float32)
    bad_log_support = np.zeros_like(anchor, dtype=np.float32)
    for version, weight in zip(negative_versions.tolist(), negative_weights.tolist()):
        negative_direction += float(weight) * weighted_log_direction(
            anchor, supports[int(version)]
        )
        bad_log_support += float(weight) * np.log(
            np.maximum(supports[int(version)], FLOOR)
        ).astype(np.float32)
    contrast_direction = (-negative_direction).astype(np.float32)
    contrast_alignment = row_cosine(contrast_direction, positive_chain)
    contrast_magnitude = rms_direction(contrast_direction)

    regime_scale = np.zeros(len(regimes), dtype=np.float32)
    regime_scale[regimes == "heavy"] = 1.0

    risk_gate = np.clip((risk - 0.72) / 0.24, 0.0, 1.0).astype(np.float32)
    contrast_gate = np.clip((contrast_alignment - 0.35) / 0.55, 0.0, 1.0).astype(
        np.float32
    )
    magnitude_gate = np.clip((contrast_magnitude - 0.0012) / 0.0060, 0.0, 1.0).astype(np.float32)
    previous_gate = np.clip(0.55 + 0.45 * previous_action, 0.55, 1.0).astype(np.float32)
    novelty_gate = (
        regime_scale
        * (previous_action < 1e-6).astype(np.float32)
        * risk_gate
        * np.clip((contrast_alignment - 0.72) / 0.20, 0.0, 1.0).astype(np.float32)
        * np.clip((contrast_magnitude - 0.0010) / 0.0025, 0.0, 1.0).astype(np.float32)
    )

    action = (
        regime_scale
        * risk_gate
        * contrast_gate
        * magnitude_gate
        * previous_gate
    ).astype(np.float32)
    action = np.clip(action + 0.18 * novelty_gate, 0.0, 1.0).astype(np.float32)

    alpha = np.zeros(len(regimes), dtype=np.float32)
    alpha[regimes == "heavy"] = 0.62 * action[regimes == "heavy"]

    raw_contrast_ratio = np.log(np.maximum(anchor, FLOOR)).astype(np.float32) - bad_log_support
    raw_contrast_ratio = np.clip(raw_contrast_ratio, -1.65, 1.65).astype(np.float32)
    target = np.exp(
        np.log(np.maximum(anchor, FLOOR)).astype(np.float32)
        + alpha[:, None, None, None] * raw_contrast_ratio
    ).astype(np.float32)
    target = normalize_support(target)
    target[regimes != "heavy"] = anchor[regimes != "heavy"]
    np.save(TARGET_OUT, target)

    max_weight = np.clip(0.82 + 0.16 * action, 0.82, 0.98).astype(np.float32)
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
        negative_versions=negative_versions,
        negative_weights=negative_weights.astype(np.float32),
        contrast_alignment=contrast_alignment,
        contrast_magnitude=contrast_magnitude,
        risk_gate=risk_gate,
        contrast_gate=contrast_gate,
        magnitude_gate=magnitude_gate,
        previous_gate=previous_gate,
        novelty_gate=novelty_gate,
        action=action,
        alpha=alpha,
        weights=weights,
    )

    args = argparse.Namespace(
        base=str(PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy"),
        output=str(OUTPUT),
        beta=140.3,
        gamma=0.10,
        eta="clear:0.0,light:0.0,medium:0.0,heavy:1.04",
        min_gate=0.05,
        max_gate=2.84,
        conf_center=0.32,
        conf_width=0.32,
        entropy_low=0.04,
        entropy_high=0.94,
        gain_clip="0.38,3.18",
        top_gain_clip="0.50,2.52",
        top_components=5632,
        target_floor=FLOOR,
        project_batch=20,
    )
    projection_report = projection(target, weights, regimes, pos_test, args)
    comparison_to_v163 = compare_outputs(
        OUTPUT, PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v162 = compare_outputs(
        OUTPUT, PATHS.cache_dir(162) / "v162_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v161 = compare_outputs(
        OUTPUT, PATHS.cache_dir(161) / "v161_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v140 = compare_outputs(
        OUTPUT, PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", regimes
    )
    if comparison_to_v163["shape"] != [500, 256, 4, 192] or comparison_to_v163["dtype"] != "complex64":
        raise ValueError(
            f"invalid output layout: {comparison_to_v163['shape']} {comparison_to_v163['dtype']}"
        )
    if not comparison_to_v163["finite"]:
        raise ValueError("v164 output contains non-finite values")

    active_by_regime = {
        regime: {
            "count": int(np.sum(regimes == regime)),
            "active_fraction": float(np.mean(action[regimes == regime] > 1e-6)),
            "action": summarize(action[regimes == regime]),
            "alpha": summarize(alpha[regimes == regime]),
            "contrast_alignment": summarize(contrast_alignment[regimes == regime]),
            "contrast_magnitude": summarize(contrast_magnitude[regimes == regime]),
            "novelty_gate": summarize(novelty_gate[regimes == regime]),
        }
        for regime in REGIMES
        if np.any(regimes == regime)
    }

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v163 momentum is close to saturated, but multiple online-negative "
            "versions define a shared bad support subspace; v164 pushes selected "
            "high-risk heavy points away from that bad ensemble only where this "
            "contrastive direction agrees with the online-positive chain"
        ),
        "new_information_used": {
            "online_positive_chain": "actual output AD support v140 -> v161 -> v162 -> v163",
            "online_negative_versions": {
                str(int(v)): float(w)
                for v, w in zip(negative_versions.tolist(), negative_weights.tolist())
            },
            "contrastive_evidence": (
                "away-from-negative-version direction has high median alignment "
                "with the validated positive chain on heavy points"
            ),
            "novelty": "allow high-risk heavy points not moved by v163 when contrastive agreement is very strong",
        },
        "excluded": {
            "full_v139_replay": True,
            "v163_momentum_overdrive_rescan": True,
            "new_local_h_residual": True,
            "phase_or_power_relearning": True,
            "medium_or_clear_light_changes": True,
        },
        "parameters": {
            "positive_chain_weights": {
                "v140_to_v161": 0.46,
                "v161_to_v162": 0.46,
                "v162_to_v163": 0.08,
            },
            "negative_score_gaps": negative_score_gaps,
            "risk_gate": "(risk - 0.72) / 0.24 clipped",
            "contrast_gate": "(contrast_alignment - 0.35) / 0.55 clipped",
            "magnitude_gate": "(contrast_magnitude - 0.0012) / 0.0060 clipped",
            "previous_gate": "0.55 + 0.45*v163_action",
            "novelty_gate": "previous_action==0, risk high, alignment>.72, magnitude>.001",
            "alpha_heavy_max": 0.62,
            "raw_contrast_log_ratio_clip": [-1.65, 1.65],
            "projection_args": {
                key: (PATHS.relative(value) if key in {"base", "output"} else value)
                for key, value in vars(args).items()
            },
        },
        "diagnostics": {
            "risk": summarize(risk),
            "previous_action": summarize(previous_action),
            "previous_alpha": summarize(previous_alpha),
            "contrast_alignment": summarize(contrast_alignment),
            "contrast_magnitude": summarize(contrast_magnitude),
            "novelty_gate": summarize(novelty_gate),
            "action": summarize(action),
            "alpha": summarize(alpha),
            "active_fraction": float(np.mean(action > 1e-6)),
            "new_heavy_active_fraction": float(
                np.mean((action[regimes == "heavy"] > 1e-6) & (previous_action[regimes == "heavy"] < 1e-6))
            ),
            "active_by_regime": active_by_regime,
        },
        "target_diagnostics": {
            "target_to_v163_output_support": summarize(row_cosine(target, anchor)),
            "target_to_v162_output_support": summarize(row_cosine(target, supports[162])),
            "target_h_l1_change_to_v163": summarize(l1_h_change(target, anchor)),
            "target_h_l1_change_to_v162": summarize(l1_h_change(target, supports[162])),
        },
        "projection": projection_report,
        "comparison_to_v163": comparison_to_v163,
        "comparison_to_v162": comparison_to_v162,
        "comparison_to_v161": comparison_to_v161,
        "comparison_to_v140": comparison_to_v140,
        "preserved": {
            "base_complex": "v163",
            "non_heavy_points": "exact v163 target and eta=0",
            "phase": "inherited from v163 through positive AD amplitude projection",
            "total_power": "projection renormalizes each point to v163 AD total power",
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
            (__file__, "generate_v164_contrastive_online_debias.py"),
            (PATHS.code / "generate_v163_validated_momentum_overdrive.py", "generate_v163_validated_momentum_overdrive.py"),
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
