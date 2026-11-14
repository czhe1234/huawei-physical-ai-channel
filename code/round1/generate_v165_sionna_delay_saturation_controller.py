#!/usr/bin/env python3
"""Generate v165: Sionna delay-consistency saturation controller.

Hypothesis:
    v163 is the current online best, but its gain over v162 is tiny.  A direct
    continuation of the same support momentum is likely saturated, and v164
    showed that away-from-negative support is not a general target generator.

    Since the dataset is Sionna-generated, the path delay implied by geometry
    is a harder physical constraint than historical version differences.  This
    version uses Sionna path-candidate delays only as a per-point saturation
    detector:

    * if v163's delay marginal is more consistent with geometric path delays
      than v162, allow a small extra v162->v163 support step;
    * if v163 is less delay-consistent than v162, roll that point partially
      back toward v162.

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
from generate_v161_hminus3_tail_distillation import l1_h_change, row_cosine
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v165_sionna_delay_saturation_controller"
OUT_DIR = PATHS.cache_dir(165)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(165)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v165_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v165_sionna_delay_saturation_target.npy"
WEIGHTS_OUT = OUT_DIR / "v165_sionna_delay_saturation_weights.npy"
ACTION_OUT = OUT_DIR / "v165_sionna_delay_saturation_actions.npz"
REPORT = REPORT_DIR / "v165_sionna_delay_saturation_generate.json"
ZIP_OUT = PATHS.round1_versions / "v165_sionna_delay_saturation_controller_candidate_pending.zip"


def support_cache(version: int) -> str:
    candidates = [
        OUT_DIR / f"probe_v{version}_collapsed_support.npy",
        PATHS.cache_dir(163) / f"probe_v{version}_collapsed_support.npy",
        PATHS.cache_dir(164) / f"probe_v{version}_collapsed_support.npy",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return str(OUT_DIR / f"probe_v{version}_collapsed_support.npy")


def path_delay_kernels() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return narrow and broad delay kernels from Sionna path candidates."""
    path_npz = np.load(PATHS.shared_features / "sionna_path_candidates.npz")
    valid = path_npz["test_valid"].astype(bool)
    path_type = path_npz["test_path_type"]
    delay = np.nan_to_num(
        np.asarray(path_npz["test_delay_bin"], dtype=np.float32),
        nan=-99.0,
        posinf=-99.0,
        neginf=-99.0,
    )
    score = np.nan_to_num(
        np.asarray(path_npz["test_score"], dtype=np.float32),
        nan=-8.0,
        posinf=4.0,
        neginf=-8.0,
    )
    mask = valid & (delay >= 0.0) & (delay < 40.0)
    weight = np.where(mask, np.exp(np.clip(score - 3.0, -8.0, 2.0)), 0.0).astype(
        np.float32
    )
    # Give the deterministic LOS slot and the top reflection family modest
    # prior mass, but keep this as a selector, not a new full support target.
    weight[:, 0] += np.where(mask[:, 0], 0.80, 0.0).astype(np.float32)
    weight += np.where(mask & (path_type == 1), 0.25, 0.0).astype(np.float32)
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), TINY)

    grid = np.arange(192, dtype=np.float32)[None, None, :]
    narrow = np.exp(-0.5 * ((grid - delay[:, :, None]) / 2.2) ** 2).astype(np.float32)
    narrow *= weight[:, :, None]
    narrow = narrow.sum(axis=1)
    narrow /= np.maximum(narrow.sum(axis=1, keepdims=True), TINY)

    broad = np.exp(-0.5 * ((grid - delay[:, :, None]) / 5.0) ** 2).astype(np.float32)
    broad *= mask[:, :, None]
    broad = broad.max(axis=1)

    meta = {
        "candidate_count": int(valid.shape[1]),
        "valid_fraction": float(np.mean(mask)),
        "los_clear_fraction": float(np.mean(path_npz["test_los_clear"].astype(bool))),
        "delay_bin": summarize(delay[mask]),
    }
    return narrow.astype(np.float32), broad.astype(np.float32), meta


def delay_scores(support: np.ndarray, narrow: np.ndarray, broad: np.ndarray) -> dict[str, np.ndarray]:
    delay_marginal = support.sum(axis=(1, 2)).astype(np.float32)
    grid = np.arange(192, dtype=np.float32)[None, :]
    return {
        "narrow_hit": np.sum(delay_marginal * narrow, axis=1).astype(np.float32),
        "broad_hit": np.sum(delay_marginal * broad, axis=1).astype(np.float32),
        "centroid": np.sum(delay_marginal * grid, axis=1).astype(np.float32),
    }


def main() -> None:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked_test = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "test_blocked"
    ].astype(np.float32)
    regimes = regime_of(blocked_test)
    risk = np.load(PATHS.cache_dir(140) / "v140_geodesic_risk.npy").astype(np.float32)

    support_paths = {
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

    narrow_kernel, broad_kernel, kernel_meta = path_delay_kernels()
    d162 = delay_scores(supports[162], narrow_kernel, broad_kernel)
    d163 = delay_scores(supports[163], narrow_kernel, broad_kernel)
    narrow_delta = (d163["narrow_hit"] - d162["narrow_hit"]).astype(np.float32)
    broad_delta = (d163["broad_hit"] - d162["broad_hit"]).astype(np.float32)

    heavy = (regimes == "heavy").astype(np.float32)
    risk_gate = np.clip((risk - 0.70) / 0.24, 0.0, 1.0).astype(np.float32)
    prior_gate = np.clip((previous_action - 0.08) / 0.64, 0.0, 1.0).astype(np.float32)

    delay_good_gate = np.clip((narrow_delta - 0.00012) / 0.00115, 0.0, 1.0).astype(
        np.float32
    )
    delay_good_gate *= np.clip((broad_delta + 0.00010) / 0.00190, 0.0, 1.0).astype(
        np.float32
    )
    delay_bad_gate = np.clip((-narrow_delta - 0.00005) / 0.00155, 0.0, 1.0).astype(
        np.float32
    )
    delay_bad_gate *= np.clip((-broad_delta + 0.00005) / 0.00300, 0.0, 1.0).astype(
        np.float32
    )

    boost_gate = heavy * risk_gate * prior_gate * delay_good_gate
    rollback_gate = heavy * risk_gate * prior_gate * delay_bad_gate

    # Positive means extra v162->v163 momentum; negative means rollback toward v162.
    signed_alpha = (0.34 * boost_gate - 0.82 * rollback_gate).astype(np.float32)
    signed_alpha = np.clip(signed_alpha, -0.82, 0.34).astype(np.float32)

    raw_momentum = np.log(np.maximum(supports[163], FLOOR)) - np.log(
        np.maximum(supports[162], FLOOR)
    )
    raw_momentum = np.clip(raw_momentum, -1.80, 1.80).astype(np.float32)
    target = np.exp(
        np.log(np.maximum(anchor, FLOOR)).astype(np.float32)
        + signed_alpha[:, None, None, None] * raw_momentum
    ).astype(np.float32)
    target = normalize_support(target)
    target[regimes != "heavy"] = anchor[regimes != "heavy"]
    np.save(TARGET_OUT, target)

    abs_action = np.clip(np.abs(signed_alpha) / 0.82, 0.0, 1.0).astype(np.float32)
    max_weight = np.clip(0.78 + 0.20 * abs_action, 0.78, 0.98).astype(np.float32)
    weights = np.empty((len(abs_action), 8), dtype=np.float32)
    weights[:, 0] = max_weight
    weights[:, 1:] = (1.0 - max_weight[:, None]) / 7.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    np.save(WEIGHTS_OUT, weights)

    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        risk=risk,
        previous_action=previous_action,
        risk_gate=risk_gate,
        prior_gate=prior_gate,
        narrow_delta=narrow_delta,
        broad_delta=broad_delta,
        delay_good_gate=delay_good_gate,
        delay_bad_gate=delay_bad_gate,
        boost_gate=boost_gate,
        rollback_gate=rollback_gate,
        signed_alpha=signed_alpha,
        weights=weights,
        d162_narrow_hit=d162["narrow_hit"],
        d163_narrow_hit=d163["narrow_hit"],
        d162_broad_hit=d162["broad_hit"],
        d163_broad_hit=d163["broad_hit"],
        d162_centroid=d162["centroid"],
        d163_centroid=d163["centroid"],
    )

    args = argparse.Namespace(
        base=str(PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy"),
        output=str(OUTPUT),
        beta=140.3,
        gamma=0.10,
        eta="clear:0.0,light:0.0,medium:0.0,heavy:0.98",
        min_gate=0.05,
        max_gate=2.78,
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
    comparison_to_v163 = compare_outputs(
        OUTPUT, PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v162 = compare_outputs(
        OUTPUT, PATHS.cache_dir(162) / "v162_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v140 = compare_outputs(
        OUTPUT, PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", regimes
    )
    if comparison_to_v163["shape"] != [500, 256, 4, 192] or comparison_to_v163["dtype"] != "complex64":
        raise ValueError(
            f"invalid output layout: {comparison_to_v163['shape']} {comparison_to_v163['dtype']}"
        )
    if not comparison_to_v163["finite"]:
        raise ValueError("v165 output contains non-finite values")

    active_by_regime = {
        regime: {
            "count": int(np.sum(regimes == regime)),
            "boost_fraction": float(np.mean(boost_gate[regimes == regime] > 1e-6)),
            "rollback_fraction": float(np.mean(rollback_gate[regimes == regime] > 1e-6)),
            "signed_alpha": summarize(signed_alpha[regimes == regime]),
            "narrow_delta": summarize(narrow_delta[regimes == regime]),
            "broad_delta": summarize(broad_delta[regimes == regime]),
        }
        for regime in REGIMES
        if np.any(regimes == regime)
    }

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "Sionna path-delay consistency can act as a physical saturation "
            "detector for v163: continue only where v163 improves delay "
            "consistency over v162, and rollback where v163 worsens it."
        ),
        "new_information_used": {
            "sionna_generation_law": "multipath delay must match feasible geometric path lengths",
            "online_state": "v163 is best but only +0.000044 over v162, so it is near saturation",
            "diagnostic": "v163 heavy delay-kernel hit is slightly worse than v162 on average",
        },
        "excluded": {
            "away_from_negative_common_subspace": True,
            "full_path_template_generation": True,
            "phase_or_power_relearning": True,
            "non_heavy_changes": True,
            "direct_pas_pdp_projection": True,
        },
        "parameters": {
            "path_kernel": kernel_meta,
            "risk_gate": "(risk - 0.70) / 0.24 clipped",
            "prior_gate": "(v163_action - 0.08) / 0.64 clipped",
            "delay_good_gate": "positive v163-v162 narrow/broad Sionna delay consistency",
            "delay_bad_gate": "negative v163-v162 narrow/broad Sionna delay consistency",
            "signed_alpha": "0.34*boost_gate - 0.82*rollback_gate",
            "raw_momentum_clip": [-1.80, 1.80],
            "projection_args": {
                key: (PATHS.relative(value) if key in {"base", "output"} else value)
                for key, value in vars(args).items()
            },
        },
        "diagnostics": {
            "risk": summarize(risk),
            "previous_action": summarize(previous_action),
            "narrow_delta": summarize(narrow_delta),
            "broad_delta": summarize(broad_delta),
            "boost_gate": summarize(boost_gate),
            "rollback_gate": summarize(rollback_gate),
            "signed_alpha": summarize(signed_alpha),
            "boost_fraction": float(np.mean(boost_gate > 1e-6)),
            "rollback_fraction": float(np.mean(rollback_gate > 1e-6)),
            "active_by_regime": active_by_regime,
        },
        "target_diagnostics": {
            "target_to_v163_support": summarize(row_cosine(target, supports[163])),
            "target_to_v162_support": summarize(row_cosine(target, supports[162])),
            "target_h_l1_change_to_v163": summarize(l1_h_change(target, supports[163])),
            "target_h_l1_change_to_v162": summarize(l1_h_change(target, supports[162])),
        },
        "projection": projection_report,
        "comparison_to_v163": comparison_to_v163,
        "comparison_to_v162": comparison_to_v162,
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
            (__file__, "generate_v165_sionna_delay_saturation_controller.py"),
            (PATHS.code / "generate_v163_validated_momentum_overdrive.py", "generate_v163_validated_momentum_overdrive.py"),
            (PATHS.code / "generate_v162_validated_tail_completion.py", "generate_v162_validated_tail_completion.py"),
            (PATHS.code / "generate_v161_hminus3_tail_distillation.py", "generate_v161_hminus3_tail_distillation.py"),
            (PATHS.code / "generate_v159_frontier_consensus_support_soup.py", "generate_v159_frontier_consensus_support_soup.py"),
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
