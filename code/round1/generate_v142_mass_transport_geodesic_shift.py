#!/usr/bin/env python3
"""Generate v142 with mass-conserving geodesic H-support transport.

v140 confirmed that geodesic-risk-conditioned h=-1/h=-2 correction improves
online score.  Its shift operator, however, is a geometric log blend between
the original and rolled support.  That operation emphasizes overlap and is not
literal energy transport.  v142 changes only this operator: the same v140
risk, thresholds, weights, projection parameters, phase, and total power are
kept, while support mass is linearly transported to the shifted bins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile

import numpy as np

from generate_v112_kernel_moe_path_support import normalize_support, row_cos
from generate_v140_balanced_hminus2_overdrive import build_risk
from generate_v87_easy_field_projection import summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v142_mass_transport_geodesic_shift"
OUT_DIR = PATHS.cache_dir(142)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(142)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v142_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v142_mass_transport_geodesic_shift_target.npy"
WEIGHTS_OUT = OUT_DIR / "v142_mass_transport_geodesic_shift_weights.npy"
TRANSPORT_OUT = OUT_DIR / "v142_mass_transport_selector.npy"
REPORT = REPORT_DIR / "v142_mass_transport_geodesic_shift_generate.json"
ZIP_OUT = PATHS.round1_versions / "v142_mass_transport_geodesic_shift_candidate_pending.zip"


def digest(path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def transport_support(
    source: np.ndarray, risk: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    target = np.array(source, dtype=np.float32, copy=True)
    fallback = risk > 0.64
    extreme = fallback & (risk > 0.90)
    h1_weight = np.where(fallback, 0.165 * risk, 0.0).astype(np.float32)
    h2_weight = np.where(extreme, 0.060 * risk, 0.0).astype(np.float32)

    if np.any(fallback):
        weight = h1_weight[fallback, None, None, None]
        shifted = np.roll(source[fallback], -1, axis=1)
        target[fallback] = (1.0 - weight) * source[fallback] + weight * shifted
    if np.any(extreme):
        weight = h2_weight[extreme, None, None, None]
        shifted = np.roll(source[extreme], -2, axis=1)
        target[extreme] = (1.0 - weight) * target[extreme] + weight * shifted

    target = np.maximum(target, 1e-12).astype(np.float32)
    target /= np.maximum(target.sum(axis=(1, 2, 3), keepdims=True), TINY)
    selector = np.column_stack(
        [risk, fallback.astype(np.float32), extreme.astype(np.float32), h1_weight, h2_weight]
    ).astype(np.float32)
    report = {
        "fallback_threshold": 0.64,
        "fallback_count": int(np.sum(fallback)),
        "fallback_fraction": float(np.mean(fallback)),
        "h1_weight": summarize(h1_weight),
        "extreme_threshold": 0.90,
        "extreme_count": int(np.sum(extreme)),
        "extreme_fraction": float(np.mean(extreme)),
        "h2_weight": summarize(h2_weight),
        "h3_enabled": False,
    }
    return target, selector, report


def main() -> None:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "test_blocked"
    ].astype(np.float32)
    regimes = regime_of(blocked)
    geo = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    names = [str(value) for value in geo["feature_names"]]
    probs = np.load(
        PATHS.cache_dir(129) / "v129_apex_badpoint_support_residual_probs.npy"
    ).astype(np.float32)
    risk = build_risk(regimes, probs, geo["test_features"].astype(np.float32), names)

    source = normalize_support(
        np.load(
            PATHS.cache_dir(129) / "v129_apex_badpoint_support_residual_target.npy"
        ),
        1e-12,
    )
    target, selector, transport_report = transport_support(source, risk)
    np.save(TARGET_OUT, target)
    np.save(TRANSPORT_OUT, selector)

    weights = np.empty((len(risk), 8), dtype=np.float32)
    weights[:, 0] = 0.10 + 0.90 * risk
    weights[:, 1:] = (1.0 - weights[:, :1]) / 7.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    np.save(WEIGHTS_OUT, weights)

    args = argparse.Namespace(
        base=str(PATHS.cache_dir(129) / "v129_Round1_Test_Channel.npy"),
        output=str(OUTPUT),
        beta=140.3,
        gamma=0.10,
        eta="clear:0.0,light:0.028,medium:0.470,heavy:0.780",
        min_gate=0.12,
        max_gate=2.95,
        conf_center=0.22,
        conf_width=0.38,
        entropy_low=0.02,
        entropy_high=0.90,
        gain_clip="0.34,3.20",
        top_gain_clip="0.48,2.35",
        top_components=5632,
        target_floor=1e-12,
        project_batch=20,
    )
    projection_report = projection(target, weights, regimes, pos_test, args)

    v140_target = normalize_support(
        np.load(
            PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_target.npy"
        ),
        1e-12,
    )
    target_to_source = row_cos(target, source)
    target_to_v140 = row_cos(target, v140_target)
    v140_to_source = row_cos(v140_target, source)
    output = np.load(OUTPUT, mmap_mode="r")
    finite = True
    for start in range(0, len(output), 10):
        block = np.asarray(output[start : start + 10])
        finite = finite and bool(
            np.isfinite(block.real).all() and np.isfinite(block.imag).all()
        )
    if not finite:
        raise ValueError("v142 output contains non-finite values")

    report_args = dict(vars(args))
    report_args["base"] = PATHS.relative(args.base)
    report_args["output"] = PATHS.relative(args.output)

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "base": "v140 online 0.501352 method family; complex base remains v129",
        "hypothesis": (
            "v140 validated the H-shift direction, but literal mass-conserving "
            "support transport should correct misregistration more faithfully "
            "than geometric overlap blending"
        ),
        "preserved_from_v140": {
            "risk_formula": True,
            "fallback_threshold": 0.64,
            "h1_weight_formula": "0.165 * risk",
            "extreme_threshold": 0.90,
            "h2_weight_formula": "0.060 * risk",
            "h3": False,
            "projection_args": report_args,
            "complex_phase": True,
            "total_power": True,
            "pol_ue_allocation": True,
        },
        "only_primary_change": (
            "geometric log blend -> linear mass-conserving circular transport"
        ),
        "transport": transport_report,
        "target_diagnostics": {
            "target_to_source": summarize(target_to_source),
            "v140_target_to_source": summarize(v140_to_source),
            "target_to_v140_target": summarize(target_to_v140),
            "by_regime": {
                regime: {
                    "count": int(np.sum(regimes == regime)),
                    "target_to_source": summarize(target_to_source[regimes == regime]),
                    "target_to_v140_target": summarize(
                        target_to_v140[regimes == regime]
                    ),
                }
                for regime in ("clear", "light", "medium", "heavy")
            },
        },
        "projection": projection_report,
        "negative_evidence": {
            "v141_splice": "nmse_to_v140=1.2607e-7; too small to recommend",
            "continuous_gain_selector": "spatial OOF Pearson <= 0.025",
            "geodesic_knn_label_accuracy": "best 0.1671 vs baseline 0.1654",
            "direction_signed_shift": "oracle sign accuracy <= 0.566",
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET_OUT),
            "weights": PATHS.relative(WEIGHTS_OUT),
            "selector": PATHS.relative(TRANSPORT_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
        "integrity": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": finite,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source_path, name in (
            (__file__, "generate_v142_mass_transport_geodesic_shift.py"),
            (
                PATHS.code / "generate_v140_balanced_hminus2_overdrive.py",
                "generate_v140_balanced_hminus2_overdrive.py",
            ),
            (
                PATHS.code / "generate_v98_path_ensemble_graph_attention_joint_ad.py",
                "generate_v98_path_ensemble_graph_attention_joint_ad.py",
            ),
            (
                PATHS.code / "generate_v87_easy_field_projection.py",
                "generate_v87_easy_field_projection.py",
            ),
            (
                PATHS.code / "generate_v112_kernel_moe_path_support.py",
                "generate_v112_kernel_moe_path_support.py",
            ),
            (PATHS.code / "powerfield_common.py", "powerfield_common.py"),
            (PATHS.code_root / "project_paths.py", "project_paths.py"),
        ):
            archive.write(source_path, name)

    report["integrity"].update(
        {
            "sha256": digest(ZIP_OUT, "sha256"),
            "md5": digest(ZIP_OUT, "md5"),
        }
    )
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {ZIP_OUT}")
    print(json.dumps(report["transport"], indent=2))
    print(json.dumps(report["target_diagnostics"], indent=2))
    print(json.dumps(report["projection"], indent=2))
    print(json.dumps(report["integrity"], indent=2))


if __name__ == "__main__":
    main()
