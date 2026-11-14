#!/usr/bin/env python3
"""Generate v139: ultra geodesic-risk multi-shift overdrive.

This is intentionally a high-risk breakthrough attempt.

Hypothesis:
    Some v129 bad points are not explained by LOS or single-bounce reflection.
    For those points, the support skeleton is not merely too weak; it is
    slightly misregistered in H-angle/delay component space.  Earlier v53/v55
    showed that AD component shifts can help.  v133 reintroduces a targeted
    H-angle support shift only for geodesic-risk points, then applies a strong
    support-only projection.  v138 pushes this confirmed v133 idea harder:

    * lower the fallback activation threshold;
    * strengthen h=-1 support shift;
    v139 pushes one step beyond v138:

    * keep the broader v138 h=-1 activation;
    * expand h=-2 from only extreme points to high-risk heavy around-corner
      points;
    * add a very weak h=-3 nudge only for ultra-risk points;
    * use slightly stronger medium/heavy support projection and wider gain caps.

Preserved:
    v129 complex phase, scalar, total power, and pol/UE allocation.
Changed:
    collapsed H/V/delay support, with aggressive risk-conditioned projection.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from generate_v87_easy_field_projection import summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, BS_POS, regime_of


VERSION = "v139_ultra_geodesic_multishift"
OUT_DIR = ASSETS / "cache_v139"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZIP_OUT = PATHS.round1_versions / "v139_ultra_geodesic_multishift_candidate_pending.zip"
REPORT = ASSETS / "v139_ultra_geodesic_multishift_generate.json"
OUTPUT = OUT_DIR / "v139_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v139_ultra_geodesic_multishift_target.npy"
WEIGHTS_OUT = OUT_DIR / "v139_ultra_geodesic_multishift_weights.npy"
SHIFT_OUT = OUT_DIR / "v139_geodesic_multishift_predictions.npy"
RISK_OUT = OUT_DIR / "v139_geodesic_risk.npy"


SHIFTS = np.arange(-4, 5, dtype=np.int32)
SHIFT_TO_INDEX = {int(s): i for i, s in enumerate(SHIFTS)}


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a.reshape(a.shape[0], -1).astype(np.float64)
    bb = b.reshape(b.shape[0], -1).astype(np.float64)
    return (np.sum(aa * bb, axis=1) / (np.sqrt(np.sum(aa * aa, axis=1) * np.sum(bb * bb, axis=1)) + 1e-30)).astype(np.float32)


def entropy01(p: np.ndarray) -> np.ndarray:
    return (-(p * np.log(np.maximum(p, 1e-12))).sum(axis=1) / math.log(p.shape[1])).astype(np.float32)


def clipped01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def support_features(s: np.ndarray) -> np.ndarray:
    n = s.shape[0]
    flat = s.reshape(n, -1)
    beam = s.sum(axis=-1).reshape(n, -1)
    delay = s.sum(axis=(1, 2))
    top = np.sort(flat, axis=1)[:, ::-1]
    parts = [
        top[:, :1],
        top[:, :4].sum(axis=1, keepdims=True),
        top[:, :16].sum(axis=1, keepdims=True),
        top[:, :64].sum(axis=1, keepdims=True),
        -(flat * np.log(flat + 1e-30)).sum(axis=1, keepdims=True) / math.log(flat.shape[1]),
        -(beam * np.log(beam + 1e-30)).sum(axis=1, keepdims=True) / math.log(beam.shape[1]),
        -(delay * np.log(delay + 1e-30)).sum(axis=1, keepdims=True) / math.log(delay.shape[1]),
        beam.max(axis=1, keepdims=True),
        delay.max(axis=1, keepdims=True),
        beam,
        delay,
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def point_features(pos: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    az = np.arctan2(pos[:, 1] - BS_POS[1], pos[:, 0] - BS_POS[0])
    return np.column_stack(
        [
            (pos[:, 0] - 150.0) / 100.0,
            (pos[:, 1] + 75.0) / 100.0,
            d / 100.0,
            np.log(d + 1e-6) - 4.0,
            np.sin(az),
            np.cos(az),
            blocked / 100.0,
            np.log1p(np.maximum(blocked, 0.0)),
        ]
    ).astype(np.float32)


def build_risk(regimes: np.ndarray, probs: np.ndarray, geo: np.ndarray, names: list[str]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(names)}
    ent = entropy01(probs)
    top = probs.max(axis=1)
    geo_ratio = geo[:, idx["geo_ratio"]]
    geo_excess = geo[:, idx["geo_excess"]]
    mean_clear = geo[:, idx["mean_clearance"]]
    turn_rate = geo[:, idx["turn_rate"]]
    turn_count = geo[:, idx["turn_count"]]

    raw = (
        0.30 * clipped01((geo_ratio - 1.13) / 0.34)
        + 0.18 * clipped01((geo_excess - 14.0) / 52.0)
        + 0.10 * clipped01((5.2 - mean_clear) / 5.2)
        + 0.10 * clipped01((turn_rate - 0.14) / 0.26)
        + 0.22 * clipped01((ent - 0.22) / 0.40)
        + 0.10 * clipped01((0.62 - top) / 0.36)
    ).astype(np.float32)
    scale_map = {"clear": 0.08, "light": 0.36, "medium": 0.92, "heavy": 1.16}
    scale = np.array([scale_map[str(r)] for r in regimes], dtype=np.float32)
    risk = clipped01(raw * scale)
    risk = clipped01((risk - 0.20) / 0.56) ** 1.05
    # Add a direct high-turn override for likely around-corner routes.
    risk = np.maximum(risk, 0.55 * clipped01((turn_count - 26.0) / 42.0) * (regimes == "heavy"))
    return clipped01(risk)


def train_shift_classifier(train_support: np.ndarray, truth_support: np.ndarray) -> tuple[ExtraTreesClassifier, dict[str, object]]:
    base_cos = row_cos(train_support, truth_support)
    cos_shift = np.stack([row_cos(np.roll(train_support, int(s), axis=1), truth_support) for s in SHIFTS], axis=1)
    best_idx = np.argmax(cos_shift, axis=1)
    best_shift = SHIFTS[best_idx]
    best_gain = cos_shift[np.arange(len(train_support)), best_idx] - base_cos
    label = best_shift.copy()
    # Aggressive: accept small train oracle gains because test projection is
    # gated later by geodesic risk and classifier confidence.
    label[best_gain < 0.006] = 0

    scene = np.load(ASSETS + "/official_pointcloud_scene_cache.npz")
    geo = np.load(ASSETS + "/geodesic_support_features.npz")
    tmpl = np.load(ASSETS + "/train_v125eq_crossfit_template_probs.npy").astype(np.float32)
    pos = np.load(str(PATHS.round1 / "Round1_Train_Pos.npy")).astype(np.float32)
    blocked = np.load(ASSETS + "/height_geometry_cache.npz")["train_blocked"].astype(np.float32)
    x = np.concatenate(
        [
            point_features(pos, blocked),
            scene["train_features"].astype(np.float32),
            geo["train_features"].astype(np.float32),
            tmpl,
            support_features(train_support),
        ],
        axis=1,
    )
    x = np.nan_to_num(x, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    y = np.array([SHIFT_TO_INDEX[int(s)] for s in label], dtype=np.int64)
    clf = ExtraTreesClassifier(
        n_estimators=900,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=13307,
        n_jobs=-1,
    )
    clf.fit(x, y)
    report = {
        "base_cos": summarize(base_cos),
        "oracle_gain": summarize(best_gain),
        "label_counts": {str(int(s)): int(np.sum(label == s)) for s in SHIFTS},
        "oracle_nonzero_frac": float(np.mean(best_shift != 0)),
        "label_nonzero_frac": float(np.mean(label != 0)),
    }
    return clf, report


def predict_shift(clf: ExtraTreesClassifier, test_support: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    scene = np.load(ASSETS + "/official_pointcloud_scene_cache.npz")
    geo = np.load(ASSETS + "/geodesic_support_features.npz")
    tmpl = np.load(ASSETS + "/cache_v129/v129_apex_badpoint_support_residual_probs.npy").astype(np.float32)
    pos = np.load(str(PATHS.round1 / "Round1_Test_Pos.npy")).astype(np.float32)
    blocked = np.load(ASSETS + "/height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    x = np.concatenate(
        [
            point_features(pos, blocked),
            scene["test_features"].astype(np.float32),
            geo["test_features"].astype(np.float32),
            tmpl,
            support_features(test_support),
        ],
        axis=1,
    )
    x = np.nan_to_num(x, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    p_small = clf.predict_proba(x)
    p = np.zeros((len(x), len(SHIFTS)), dtype=np.float32)
    for j, c in enumerate(clf.classes_):
        p[:, int(c)] = p_small[:, j]
    pred_idx = np.argmax(p, axis=1)
    pred_shift = SHIFTS[pred_idx].astype(np.int32)
    conf = p[np.arange(len(x)), pred_idx].astype(np.float32)
    report = {
        "pred_counts_raw": {str(int(s)): int(np.sum(pred_shift == s)) for s in SHIFTS},
        "pred_conf": summarize(conf),
        "nonzero_raw_frac": float(np.mean(pred_shift != 0)),
    }
    return pred_shift, conf, p, report


def shifted_log_blend(target: np.ndarray, pred_shift: np.ndarray, shift_weight: np.ndarray, risk: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    out = np.array(target, dtype=np.float32, copy=True)
    floor = 1e-12
    active = np.zeros(len(target), dtype=bool)
    for s in SHIFTS:
        if int(s) == 0:
            continue
        m = pred_shift == int(s)
        if not np.any(m):
            continue
        shifted = np.roll(target[m], int(s), axis=1).astype(np.float32)
        w = shift_weight[m].astype(np.float32)[:, None, None, None]
        out[m] = np.exp((1.0 - w) * np.log(np.maximum(target[m], floor)) + w * np.log(np.maximum(shifted, floor))).astype(np.float32)
        active[m] = True
    # Around-corner fallback: keep the v138 broad h=-1 bias and push it
    # slightly stronger.
    fallback = (~active) & (risk > 0.62)
    if np.any(fallback):
        shifted = np.roll(target[fallback], -1, axis=1).astype(np.float32)
        w = (0.18 * risk[fallback]).astype(np.float32)[:, None, None, None]
        out[fallback] = np.exp((1.0 - w) * np.log(np.maximum(target[fallback], floor)) + w * np.log(np.maximum(shifted, floor))).astype(np.float32)
        active[fallback] = True
        pred_shift[fallback] = -1
        shift_weight[fallback] = np.maximum(shift_weight[fallback], (0.18 * risk[fallback]).astype(np.float32))
    # High-risk circuitous points get a stronger h=-2 correction than v138.
    # This explicitly tests whether v138's positive result was still
    # under-shifting the hardest around-corner points.
    extreme = active & (risk > 0.86)
    if np.any(extreme):
        shifted2 = np.roll(target[extreme], -2, axis=1).astype(np.float32)
        w2 = (0.080 * risk[extreme]).astype(np.float32)[:, None, None, None]
        out[extreme] = np.exp((1.0 - w2) * np.log(np.maximum(out[extreme], floor)) + w2 * np.log(np.maximum(shifted2, floor))).astype(np.float32)
        pred_shift[extreme] = -2
        shift_weight[extreme] = np.maximum(shift_weight[extreme], (0.18 * risk[extreme] + 0.080 * risk[extreme]).astype(np.float32))
    # Ultra-risk points get a tiny h=-3 tail.  This is deliberately weak to
    # avoid erasing the v138 skeleton while still probing a longer path shift.
    ultra = active & (risk > 0.985)
    if np.any(ultra):
        shifted3 = np.roll(target[ultra], -3, axis=1).astype(np.float32)
        w3 = (0.025 * risk[ultra]).astype(np.float32)[:, None, None, None]
        out[ultra] = np.exp((1.0 - w3) * np.log(np.maximum(out[ultra], floor)) + w3 * np.log(np.maximum(shifted3, floor))).astype(np.float32)
        pred_shift[ultra] = -3
        shift_weight[ultra] = np.maximum(shift_weight[ultra], (0.18 * risk[ultra] + 0.080 * risk[ultra] + 0.025 * risk[ultra]).astype(np.float32))
    out /= np.maximum(out.sum(axis=(1, 2, 3), keepdims=True), TINY)
    report = {
        "active_frac": float(np.mean(active)),
        "fallback_frac": float(np.mean(fallback)),
        "extreme_frac": float(np.mean(extreme)),
        "ultra_frac": float(np.mean(ultra)),
        "shift_weight": summarize(shift_weight),
        "final_counts": {str(int(s)): int(np.sum((pred_shift == s) & active)) for s in SHIFTS},
    }
    return out.astype(np.float32), report


def main() -> None:
    pos_test = np.load(str(PATHS.round1 / "Round1_Test_Pos.npy")).astype(np.float64)
    blocked_test = np.load(ASSETS + "/height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked_test)

    # Train-side support for shift oracle.
    truth = np.load(ASSETS + "/truth_coll_power.npy").astype(np.float32)
    truth /= np.maximum(truth.sum(axis=(1, 2, 3), keepdims=True), TINY)
    train_base = np.load(ASSETS + "/cache_v128/train_v125eq_collapsed_support.npy").astype(np.float32)
    train_base /= np.maximum(train_base.sum(axis=(1, 2, 3), keepdims=True), TINY)

    clf, clf_report = train_shift_classifier(train_base, truth)

    target = np.load(ASSETS + "/cache_v129/v129_apex_badpoint_support_residual_target.npy").astype(np.float32)
    target /= np.maximum(target.sum(axis=(1, 2, 3), keepdims=True), TINY)

    geo = np.load(ASSETS + "/geodesic_support_features.npz")
    names = [str(x) for x in geo["feature_names"]]
    probs = np.load(ASSETS + "/cache_v129/v129_apex_badpoint_support_residual_probs.npy").astype(np.float32)
    risk = build_risk(regimes, probs, geo["test_features"].astype(np.float32), names)

    pred_shift, conf, prob, pred_report = predict_shift(clf, target)
    # Aggressive activation.  Nonzero shift class is rare, so use low threshold
    # but require geodesic risk support.
    nonzero_prob = 1.0 - prob[:, SHIFT_TO_INDEX[0]]
    active = (pred_shift != 0) & (nonzero_prob > 0.16) & (risk > 0.20)
    pred_shift = np.where(active, pred_shift, 0).astype(np.int32)
    shift_weight = clipped01((nonzero_prob - 0.14) / 0.42) * clipped01((risk - 0.12) / 0.70)
    shift_weight = np.where(active, np.minimum(0.42, shift_weight), 0.0).astype(np.float32)

    final_target, shift_report = shifted_log_blend(target, pred_shift.copy(), shift_weight.copy(), risk)
    np.save(TARGET_OUT, final_target)
    np.save(SHIFT_OUT, np.column_stack([pred_shift, conf, nonzero_prob, shift_weight, risk]).astype(np.float32))
    np.save(RISK_OUT, risk.astype(np.float32))

    weights = np.empty((len(risk), 8), dtype=np.float32)
    weights[:, 0] = 0.10 + 0.90 * risk
    weights[:, 1:] = (1.0 - weights[:, :1]) / 7.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    np.save(WEIGHTS_OUT, weights)

    args = argparse.Namespace(
        base=ASSETS + "/cache_v129/v129_Round1_Test_Channel.npy",
        output=str(OUTPUT),
        beta=140.3,
        gamma=0.10,
        eta="clear:0.0,light:0.030,medium:0.500,heavy:0.840",
        min_gate=0.12,
        max_gate=3.10,
        conf_center=0.22,
        conf_width=0.38,
        entropy_low=0.02,
        entropy_high=0.90,
        gain_clip="0.30,3.45",
        top_gain_clip="0.42,2.55",
        top_components=6144,
        target_floor=1e-12,
        project_batch=20,
    )
    proj = projection(final_target, weights, regimes, pos_test, args)

    report = {
        "method": VERSION,
        "base": "v129 online 0.498117; v133 online 0.499218; v138 online 0.500793",
        "operation": (
            "ultra version of v138: broader/stronger geodesic-risk h=-1 "
            "support fallback, expanded h=-2 nudge for high-risk points, "
            "tiny h=-3 nudge for ultra-risk points, and stronger medium/heavy "
            "support-only projection"
        ),
        "args": vars(args),
        "risk": {
            "risk": summarize(risk),
            "nonzero_frac": float(np.mean(risk > 0)),
            "high_frac": float(np.mean(risk > 0.55)),
            "by_regime": {r: summarize(risk[regimes == r]) if np.any(regimes == r) else {} for r in ("clear", "light", "medium", "heavy")},
        },
        "shift_classifier": clf_report,
        "shift_prediction": pred_report,
        "shift_application": shift_report,
        "projection": proj,
        "outputs": {
            "submission_npy": str(OUTPUT),
            "target": str(TARGET_OUT),
            "weights": str(WEIGHTS_OUT),
            "shift_predictions": str(SHIFT_OUT),
            "risk": str(RISK_OUT),
            "zip": str(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(OUTPUT, "Round1_Test_Channel.npy")
        for src, arc in [
            (__file__, "generate_v139_ultra_geodesic_multishift.py"),
            (ASSETS + "/generate_v138_extreme_geodesic_shift_overdrive.py", "generate_v138_extreme_geodesic_shift_overdrive.py"),
            (ASSETS + "/generate_v133_geodesic_shift_overdrive.py", "generate_v133_geodesic_shift_overdrive.py"),
            (ASSETS + "/generate_v132_geodesic_risk_apex.py", "generate_v132_geodesic_risk_apex.py"),
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
