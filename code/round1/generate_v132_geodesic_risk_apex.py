#!/usr/bin/env python3
"""Generate v132: geodesic-risk gated apex support projection.

v131 showed that family residual cannot directly take over the support target.
The geodesic analysis, however, found that soft point-cloud corridor cost is a
useful bad-point risk signal.  v132 therefore uses geodesic/corridor topology
only as an extra selector:

* base complex channel: v129, current online best;
* target support: v129 support target, not a new noisy family residual;
* selector: high geodesic excess/ratio, low path clearance, high template
  uncertainty, and medium/heavy regime;
* operation: tiny support-only overdrive preserving v129 phase, scalar,
  total power, and pol/UE allocation.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from project_paths import PATHS
from powerfield_common import ASSETS, regime_of


VERSION = "v132_geodesic_risk_apex"
OUT_DIR = ASSETS / "cache_v132"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZIP_OUT = PATHS.round1_versions / "v132_geodesic_risk_apex_candidate_pending.zip"
REPORT = ASSETS / "v132_geodesic_risk_apex_generate.json"
OUTPUT = OUT_DIR / "v132_Round1_Test_Channel.npy"
WEIGHTS_OUT = OUT_DIR / "v132_geodesic_risk_weights.npy"
RISK_OUT = OUT_DIR / "v132_geodesic_risk.npy"


REGIME_SCALE = {
    "clear": 0.00,
    "light": 0.22,
    "medium": 0.78,
    "heavy": 1.00,
}


def entropy01(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    return (-(p * np.log(np.maximum(p, 1e-12))).sum(axis=1) / math.log(p.shape[1])).astype(np.float32)


def clipped01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def build_risk() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    feat_npz = np.load(ASSETS + "/geodesic_support_features.npz")
    geo = feat_npz["test_features"].astype(np.float32)
    names = [str(x) for x in feat_npz["feature_names"]]
    idx = {n: i for i, n in enumerate(names)}

    blocked = np.load(ASSETS + "/height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    probs = np.load(ASSETS + "/cache_v129/v129_apex_badpoint_support_residual_probs.npy").astype(np.float32)
    ent = entropy01(probs)
    top = probs.max(axis=1)

    geo_ratio = geo[:, idx["geo_ratio"]]
    geo_excess = geo[:, idx["geo_excess"]]
    mean_clear = geo[:, idx["mean_clearance"]]
    turn_rate = geo[:, idx["turn_rate"]]
    turn_count = geo[:, idx["turn_count"]]

    risk_geo = clipped01((geo_ratio - 1.18) / 0.28)
    risk_excess = clipped01((geo_excess - 18.0) / 42.0)
    risk_clear = clipped01((4.8 - mean_clear) / 4.8)
    risk_turn = clipped01((turn_rate - 0.18) / 0.20)
    risk_entropy = clipped01((ent - 0.25) / 0.35)
    risk_low_top = clipped01((0.58 - top) / 0.32)

    raw = (
        0.30 * risk_geo
        + 0.20 * risk_excess
        + 0.12 * risk_clear
        + 0.08 * risk_turn
        + 0.20 * risk_entropy
        + 0.10 * risk_low_top
    ).astype(np.float32)
    scale = np.array([REGIME_SCALE[str(r)] for r in regimes], dtype=np.float32)
    risk = clipped01(raw * scale)
    # Keep this as a precision selector, not a broad new target.
    risk = clipped01((risk - 0.34) / 0.46) ** 1.35

    weights = np.empty((len(risk), 8), dtype=np.float32)
    weights[:, 0] = 0.125 + 0.875 * risk
    weights[:, 1:] = (1.0 - weights[:, :1]) / 7.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    report = {
        "risk": summarize(risk),
        "risk_nonzero_frac": float(np.mean(risk > 0.0)),
        "risk_high_frac": float(np.mean(risk > 0.5)),
        "geo_ratio": summarize(geo_ratio),
        "geo_excess": summarize(geo_excess),
        "mean_clearance": summarize(mean_clear),
        "template_entropy": summarize(ent),
        "template_top_prob": summarize(top),
        "turn_count": summarize(turn_count),
        "by_regime": {
            r: {
                "count": int(np.sum(regimes == r)),
                "risk": summarize(risk[regimes == r]) if np.any(regimes == r) else {},
            }
            for r in ("clear", "light", "medium", "heavy")
        },
    }
    return risk.astype(np.float32), weights.astype(np.float32), report


def main() -> None:
    risk, weights, risk_report = build_risk()
    np.save(RISK_OUT, risk)
    np.save(WEIGHTS_OUT, weights)

    pos_test = np.load(str(PATHS.round1 / "Round1_Test_Pos.npy")).astype(np.float64)
    blocked = np.load(ASSETS + "/height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    target = np.load(ASSETS + "/cache_v129/v129_apex_badpoint_support_residual_target.npy", mmap_mode="r")

    args = argparse.Namespace(
        base=ASSETS + "/cache_v129/v129_Round1_Test_Channel.npy",
        output=str(OUTPUT),
        beta=140.3,
        gamma=0.10,
        eta="clear:0.0,light:0.001,medium:0.085,heavy:0.135",
        min_gate=0.0,
        max_gate=1.0,
        conf_center=0.38,
        conf_width=0.34,
        entropy_low=0.04,
        entropy_high=0.82,
        gain_clip="0.74,1.44",
        top_gain_clip="0.80,1.34",
        top_components=1792,
        target_floor=1e-12,
        project_batch=20,
    )
    proj = projection(target, weights, regimes, pos_test, args)

    report = {
        "method": VERSION,
        "base": "v129 online 0.498117",
        "operation": (
            "geodesic/corridor risk used only as selector for a tiny extra "
            "projection toward the v129 support target; preserve v129 phase, "
            "scalar, total power, and pol/UE allocation"
        ),
        "args": vars(args),
        "risk": risk_report,
        "projection": proj,
        "outputs": {
            "submission_npy": str(OUTPUT),
            "risk": str(RISK_OUT),
            "weights": str(WEIGHTS_OUT),
            "zip": str(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(OUTPUT, "Round1_Test_Channel.npy")
        for src, arc in [
            (__file__, "generate_v132_geodesic_risk_apex.py"),
            (ASSETS + "/analyze_geodesic_support_source.py", "analyze_geodesic_support_source.py"),
            (ASSETS + "/generate_v98_path_ensemble_graph_attention_joint_ad.py", "generate_v98_path_ensemble_graph_attention_joint_ad.py"),
            (ASSETS + "/generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
            (ASSETS + "/powerfield_common.py", "powerfield_common.py"),
            (PATHS.code_root / "project_paths.py", "project_paths.py"),
        ]:
            zf.write(src, arc)
    print(f"wrote {ZIP_OUT}", flush=True)


if __name__ == "__main__":
    main()
