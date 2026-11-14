#!/usr/bin/env python3
"""Generate v159: online-frontier consensus support soup.

Hypothesis:
    v143, v150 and v152 are all very close to the current online best v140,
    but they use different extra information: test-test geodesic risk
    consistency, a LOS-safe continuous local action controller, and a
    reverse-source local H field.  Individually each one is slightly below
    v140, so their raw corrections are not reliable enough to replace v140.
    Their shared AD-support direction, however, may be a useful weak
    consensus signal.

Changed:
    medium/heavy collapsed joint angle-delay support is nudged toward a
    geometric consensus of v143/v150/v152, with v140 kept as the base anchor.

Preserved:
    v140 complex phase, point total power, clear points, and pol/UE allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import (
    BS_POS,
    ad_transform,
    beta_subcarriers,
    row_cos,
    summarize,
)
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v159_frontier_consensus_support_soup"
OUT_DIR = PATHS.cache_dir(159)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(159)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v159_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v159_frontier_consensus_target.npy"
CONSENSUS_OUT = OUT_DIR / "v159_frontier_consensus_alt_support.npy"
WEIGHTS_OUT = OUT_DIR / "v159_frontier_consensus_weights.npy"
ACTION_OUT = OUT_DIR / "v159_frontier_consensus_actions.npz"
REPORT = REPORT_DIR / "v159_frontier_consensus_support_soup_generate.json"
ZIP_OUT = PATHS.round1_versions / "v159_frontier_consensus_support_soup_candidate_pending.zip"

FLOOR = 1e-12
REGIMES = ("clear", "light", "medium", "heavy")

FRONTIER = {
    "v140": {
        "score": 0.501352,
        "path": PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy",
        "role": "online_best_anchor",
    },
    "v143": {
        "score": 0.501271,
        "path": PATHS.cache_dir(143) / "v143_Round1_Test_Channel.npy",
        "role": "test_test_geodesic_risk_consistency",
    },
    "v150": {
        "score": 0.501270,
        "path": PATHS.cache_dir(150) / "v150_Round1_Test_Channel.npy",
        "role": "los_safe_continuous_action_controller",
    },
    "v152": {
        "score": 0.501281,
        "path": PATHS.cache_dir(152) / "v152_Round1_Test_Channel.npy",
        "role": "reverse_source_local_h_field",
    },
}


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def normalize_support(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return out.astype(np.float32)


def collapsed_support_from_output(
    path: Path,
    positions: np.ndarray,
    cache_path: Path,
    batch: int = 20,
) -> np.ndarray:
    if cache_path.is_file():
        print(f"reuse support cache {PATHS.relative(cache_path)}", flush=True)
        return normalize_support(np.load(cache_path).astype(np.float32))

    source = np.load(path, mmap_mode="r")
    beta_s = beta_subcarriers(140.3, 0.10)
    distance = np.linalg.norm(positions - BS_POS[None, :], axis=1)
    support = np.empty((source.shape[0], 16, 8, 192), dtype=np.float32)
    for start in range(0, source.shape[0], batch):
        end = min(start + batch, source.shape[0])
        h = np.asarray(source[start:end], dtype=np.complex64)
        derotated = h * np.exp(
            1j * distance[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        ad = ad_transform(derotated)
        power = (np.abs(ad) ** 2).astype(np.float32)
        collapsed = power.sum(axis=(1, 4))
        support[start:end] = collapsed / np.maximum(
            collapsed.sum(axis=(1, 2, 3), keepdims=True), TINY
        )
        print(f"{path.name} support {end}/{source.shape[0]}", flush=True)
    support = normalize_support(support)
    np.save(cache_path, support)
    return support


def geometric_mix(supports: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    weights /= np.maximum(weights.sum(), TINY)
    log_acc = np.zeros_like(supports[0], dtype=np.float32)
    for support, weight in zip(supports, weights):
        log_acc += float(weight) * np.log(np.maximum(support, FLOOR)).astype(np.float32)
    return normalize_support(np.exp(log_acc).astype(np.float32))


def compare_outputs(output_path: Path, base_path: Path, regimes: np.ndarray) -> dict[str, object]:
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
        numerator[start:end] = np.sum(
            np.abs(current - reference) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        denominator[start:end] = np.sum(np.abs(reference) ** 2, axis=(1, 2, 3), dtype=np.float64)
        out_power[start:end] = np.sum(np.abs(current) ** 2, axis=(1, 2, 3), dtype=np.float64)
        base_power[start:end] = denominator[start:end]
        finite = finite and bool(np.isfinite(current.real).all() and np.isfinite(current.imag).all())
    point_nmse = numerator / np.maximum(denominator, TINY)
    return {
        "finite": finite,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "nmse_to_v140": float(numerator.sum() / max(denominator.sum(), TINY)),
        "nmse_to_v140_by_regime": {
            regime: float(numerator[regimes == regime].sum() / max(denominator[regimes == regime].sum(), TINY))
            for regime in REGIMES
            if np.any(regimes == regime)
        },
        "point_nmse": summarize(point_nmse),
        "power_ratio_to_v140": summarize(out_power / np.maximum(base_power, TINY)),
    }


def main() -> None:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked_test = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "test_blocked"
    ].astype(np.float32)
    regimes = regime_of(blocked_test)
    risk = np.load(PATHS.cache_dir(140) / "v140_geodesic_risk.npy").astype(np.float32)

    supports: dict[str, np.ndarray] = {}
    for name, item in FRONTIER.items():
        support_cache = OUT_DIR / f"{name}_collapsed_support.npy"
        supports[name] = collapsed_support_from_output(item["path"], pos_test, support_cache)

    anchor = supports["v140"]
    alt_names = ["v143", "v150", "v152"]
    alt_weights = np.asarray([0.30, 0.24, 0.46], dtype=np.float32)
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

    # Move only when the non-v140 frontier versions agree with each other and
    # actually disagree with v140 enough to carry new information.
    agreement_gate = np.clip((agreement - 0.965) / 0.030, 0.0, 1.0).astype(np.float32)
    divergence_gate = np.clip((divergence - 0.0020) / 0.055, 0.0, 1.0).astype(np.float32)
    risk_gate = np.clip((risk - 0.24) / 0.70, 0.0, 1.0).astype(np.float32)
    regime_scale = np.zeros(len(regimes), dtype=np.float32)
    regime_scale[regimes == "medium"] = 0.58
    regime_scale[regimes == "heavy"] = 1.00
    action = np.clip(agreement_gate * divergence_gate * risk_gate * regime_scale, 0.0, 1.0)

    # Stronger than a passive model soup, but still anchored to v140.  The
    # exponent is kept below 0.50 because every alt version is individually
    # slightly worse online.
    alpha = np.zeros(len(regimes), dtype=np.float32)
    alpha[regimes == "medium"] = 0.26 * action[regimes == "medium"]
    alpha[regimes == "heavy"] = 0.42 * action[regimes == "heavy"]

    target = np.exp(
        (1.0 - alpha[:, None, None, None]) * np.log(np.maximum(anchor, FLOOR))
        + alpha[:, None, None, None] * np.log(np.maximum(alt_consensus, FLOOR))
    ).astype(np.float32)
    target = normalize_support(target)
    target[regimes == "clear"] = anchor[regimes == "clear"]
    np.save(TARGET_OUT, target)

    max_weight = np.clip(0.20 + 0.76 * action, 0.20, 0.96).astype(np.float32)
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
        divergence_gate=divergence_gate,
        risk_gate=risk_gate,
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
        eta="clear:0.0,light:0.0,medium:0.34,heavy:0.62",
        min_gate=0.08,
        max_gate=2.20,
        conf_center=0.24,
        conf_width=0.44,
        entropy_low=0.04,
        entropy_high=0.92,
        gain_clip="0.45,2.80",
        top_gain_clip="0.55,2.20",
        top_components=5632,
        target_floor=FLOOR,
        project_batch=20,
    )
    projection_report = projection(target, weights, regimes, pos_test, args)
    comparison = compare_outputs(OUTPUT, FRONTIER["v140"]["path"], regimes)
    if comparison["shape"] != [500, 256, 4, 192] or comparison["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison['shape']} {comparison['dtype']}")
    if not comparison["finite"]:
        raise ValueError("v159 output contains non-finite values")

    target_cos = row_cos(target, anchor)
    alt_target_cos = row_cos(target, alt_consensus)
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
            "online-near-best versions contain complementary weak support "
            "signals; their shared AD-support direction can improve v140 "
            "more safely than train-proxy residuals or direct path peak priors"
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
            "alpha_medium_max": 0.26,
            "alpha_heavy_max": 0.42,
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
            "target_to_alt_consensus": summarize(alt_target_cos),
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
            "clear_points": "exact v140 target and eta=0",
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
            (__file__, "generate_v159_frontier_consensus_support_soup.py"),
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
