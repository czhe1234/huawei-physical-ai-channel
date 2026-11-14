#!/usr/bin/env python3
"""Generate v120: consensus component selector apex.

This is an aggressive post-v119 attempt.

The previous large jump came from changing the optimization object from full
complex H to collapsed AD support.  v119 showed that full per-pol/per-UE block
support contains signal but is too noisy when projected directly.  v120 uses
that lesson as a selector, not as a direct target.

The idea:

* base complex channel: v115, current online best;
* proven support skeleton: v115/v112/v111/v109;
* aggressive but risky sources: v110, v113, v118, v119;
* build a component-level consensus direction only where several risky sources
  agree and the proven skeleton does not reject it;
* push medium/heavy regions hard, but preserve phase, scalar structure, and
  per-point total power.

No Sionna mesh/object/material labels are used by this script.  It consumes
previously generated support targets and the official point-cloud-derived v115
base.
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
from generate_v112_kernel_moe_path_support import normalize_support, row_cos
from project_paths import PATHS
from powerfield_common import ASSETS, regime_of


REGIMES = ("clear", "light", "medium", "heavy")


def log_blend(arrays: list[np.ndarray], coeff: np.ndarray, floor: float, batch: int) -> np.ndarray:
    n = arrays[0].shape[0]
    out = np.empty_like(arrays[0], dtype=np.float32)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        acc = np.zeros_like(arrays[0][start:end], dtype=np.float64)
        for j, arr in enumerate(arrays):
            c = coeff[start:end, j]
            acc += c.reshape((-1, 1, 1, 1)) * np.log(np.maximum(arr[start:end], floor))
        val = np.exp(acc).astype(np.float32)
        val /= np.maximum(val.sum(axis=(1, 2, 3), keepdims=True), 1e-30)
        out[start:end] = val
    return out


def synthetic_weights(conf: np.ndarray, cols: int, floor: float = 0.018) -> np.ndarray:
    conf = np.clip(conf.astype(np.float32), 0.0, 1.0)
    maxw = np.clip(0.18 + 0.72 * conf, 0.16, 0.90).astype(np.float32)
    w = np.empty((len(conf), cols), dtype=np.float32)
    w[:, 0] = maxw
    w[:, 1:] = (1.0 - maxw[:, None]) / float(cols - 1)
    w = np.maximum(w, floor).astype(np.float32)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-30)
    return w


def build_consensus_target(
    current: np.ndarray,
    proven: list[np.ndarray],
    risky: list[np.ndarray],
    regimes: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build a component-level target by gated support-ratio consensus."""

    n = current.shape[0]
    flat_shape = (n, -1)
    cur = current.reshape(flat_shape).astype(np.float32)
    proven_flat = [x.reshape(flat_shape).astype(np.float32) for x in proven]
    risky_flat = [x.reshape(flat_shape).astype(np.float32) for x in risky]
    floor = float(args.floor)

    # Proven skeleton: conservative geometric mean.  If this disagrees with a
    # risky component direction, damp that direction.
    skel_coeff = np.asarray([0.44, 0.28, 0.18, 0.10], dtype=np.float32)
    skel_log = np.zeros_like(cur, dtype=np.float32)
    for c, arr in zip(skel_coeff, proven_flat):
        skel_log += c * np.log(np.maximum(arr, floor))
    skeleton = np.exp(skel_log).astype(np.float32)
    skeleton /= np.maximum(skeleton.sum(axis=1, keepdims=True), 1e-30)

    # Risky sources are deliberately diverse:
    # v110: apex momentum support, never overdriven as hard as v113.
    # v113: failed overdrive, useful only where it agrees with others.
    # v118: inverse-RT selective target.
    # v119: leaked/full-block-aware collapsed target, failed directly but useful as a selector.
    risky_w = np.asarray([0.30, 0.14, 0.34, 0.22], dtype=np.float32)
    log_ratio = []
    for arr in risky_flat:
        log_ratio.append(np.log(np.maximum(arr, floor)) - np.log(np.maximum(cur, floor)))
    lr = np.stack(log_ratio, axis=0).astype(np.float32)  # R,N,D
    direction = np.tensordot(risky_w, lr, axes=(0, 0)).astype(np.float32)

    # Component-level agreement: sources must point in similar directions.
    sign_vote = np.tensordot(risky_w, np.sign(lr), axes=(0, 0)).astype(np.float32)
    abs_vote = np.tensordot(risky_w, np.abs(lr) > float(args.component_vote_threshold), axes=(0, 0)).astype(np.float32)
    agreement = np.clip(np.abs(sign_vote), 0.0, 1.0)

    # Skeleton agreement: the conservative support skeleton should not strongly
    # oppose the proposed ratio.  This is where v119/full-block becomes a
    # selector rather than a direct projection.
    skel_ratio = np.log(np.maximum(skeleton, floor)) - np.log(np.maximum(cur, floor))
    skel_ok = 1.0 / (1.0 + np.exp(-float(args.skeleton_slope) * (direction * skel_ratio + float(args.skeleton_bias))))

    # Only modify components that are relevant to the current support or are
    # repeatedly proposed by the risky sources.  This avoids full-map smearing.
    cur_rank = np.argsort(np.argsort(-cur, axis=1), axis=1).astype(np.int32)
    top_gate = (cur_rank < int(args.base_top_components)).astype(np.float32)
    risky_max = np.maximum.reduce([arr for arr in risky_flat])
    risky_rank = np.argsort(np.argsort(-risky_max, axis=1), axis=1).astype(np.int32)
    proposal_gate = (risky_rank < int(args.proposal_top_components)).astype(np.float32)
    component_gate = np.maximum(top_gate, proposal_gate)

    # Point-level trust: use high-risk directions only when they are not too far
    # from the best verified targets, but are novel enough to matter.
    risky_geo = np.exp(np.tensordot(risky_w, [np.log(np.maximum(x, floor)) for x in risky_flat], axes=(0, 0))).astype(np.float32)
    risky_geo /= np.maximum(risky_geo.sum(axis=1, keepdims=True), 1e-30)
    skeleton_cos = row_cos(risky_geo.reshape(current.shape), skeleton.reshape(current.shape)).astype(np.float32)
    current_cos = row_cos(risky_geo.reshape(current.shape), current).astype(np.float32)
    v115_v112_cos = row_cos(current, proven[1]).astype(np.float32)
    novelty = np.clip((1.0 - current_cos - float(args.novel_low)) / max(float(args.novel_high) - float(args.novel_low), 1e-6), 0.0, 1.0)
    anchor_ok = np.clip((skeleton_cos - float(args.anchor_low)) / max(float(args.anchor_high) - float(args.anchor_low), 1e-6), 0.0, 1.0)
    base_ok = np.clip((v115_v112_cos - float(args.base_low)) / max(float(args.base_high) - float(args.base_low), 1e-6), 0.0, 1.0)
    point_trust = np.clip(0.45 * anchor_ok + 0.35 * novelty + 0.20 * base_ok, 0.0, 1.0)

    reg_mult = np.asarray(
        [
            float(args.reg_clear)
            if r == "clear"
            else float(args.reg_light)
            if r == "light"
            else float(args.reg_medium)
            if r == "medium"
            else float(args.reg_heavy)
            for r in regimes
        ],
        dtype=np.float32,
    )
    point_strength = np.clip(reg_mult * (0.20 + 0.80 * point_trust), 0.0, float(args.max_strength)).astype(np.float32)

    comp_strength = (
        point_strength[:, None]
        * component_gate
        * np.clip((agreement - float(args.agreement_low)) / max(float(args.agreement_high) - float(args.agreement_low), 1e-6), 0.0, 1.0)
        * np.clip(abs_vote / float(args.vote_norm), 0.0, 1.0)
        * skel_ok
    ).astype(np.float32)
    direction = np.clip(direction, -float(args.log_ratio_clip), float(args.log_ratio_clip))
    ratio = np.exp(comp_strength * direction).astype(np.float32)

    # A small global support-template pull keeps the aggressive component move
    # from becoming too jagged.
    template_coeff = np.zeros((n, 3), dtype=np.float32)
    template_mix = np.clip(float(args.template_mix) * point_strength, 0.0, float(args.template_mix_cap))
    template_coeff[:, 0] = 1.0 - template_mix
    template_coeff[:, 1] = 0.70 * template_mix
    template_coeff[:, 2] = 0.30 * template_mix
    template = log_blend(
        [current, risky_geo.reshape(current.shape), skeleton.reshape(current.shape)],
        template_coeff,
        floor,
        int(args.batch),
    ).reshape(flat_shape)

    target = cur * ratio
    target = (1.0 - template_mix[:, None]) * target + template_mix[:, None] * template
    target = np.maximum(target, floor).astype(np.float32)
    target /= np.maximum(target.sum(axis=1, keepdims=True), 1e-30)

    # The component selector above is safe but can be too timid.  Add a direct
    # high-risk path-component template pull toward v118 where the skeleton and
    # v119 metric selector do not reject it.  This is the actual "apex" part:
    # v118 supplies the new path/component hypothesis; v119 only decides where
    # that hypothesis is not obviously inconsistent with metric-aligned support.
    v118_flat = risky_flat[2]
    v119_flat = risky_flat[3]
    v118_vs_v119 = row_cos(risky[2], risky[3]).astype(np.float32)
    metric_ok = np.clip((v118_vs_v119 - float(args.metric_low)) / max(float(args.metric_high) - float(args.metric_low), 1e-6), 0.0, 1.0)
    direct_mix = np.clip(
        float(args.inverse_direct_mix) * point_strength * (0.35 + 0.65 * anchor_ok) * (0.35 + 0.65 * metric_ok),
        0.0,
        float(args.inverse_direct_cap),
    ).astype(np.float32)
    metric_mix = np.clip(float(args.metric_template_mix) * point_strength * metric_ok, 0.0, float(args.metric_template_cap)).astype(
        np.float32
    )
    keep = np.clip(1.0 - direct_mix - metric_mix, 0.05, 1.0).astype(np.float32)
    scale = keep + direct_mix + metric_mix
    keep /= scale
    direct_mix /= scale
    metric_mix /= scale
    blended_log = (
        keep[:, None] * np.log(np.maximum(target, floor))
        + direct_mix[:, None] * np.log(np.maximum(v118_flat, floor))
        + metric_mix[:, None] * np.log(np.maximum(v119_flat, floor))
    )
    target = np.exp(blended_log).astype(np.float32)
    target /= np.maximum(target.sum(axis=1, keepdims=True), 1e-30)
    target = target.reshape(current.shape).astype(np.float32)

    conf = np.clip(point_strength / max(float(args.max_strength), 1e-6), 0.0, 1.0)
    conf *= np.clip((row_cos(target, current) - float(args.target_min_cos)) / max(1.0 - float(args.target_min_cos), 1e-6), 0.0, 1.0)
    weights = synthetic_weights(conf, int(args.weight_cols))

    report = {
        "point_strength": summarize(point_strength),
        "point_trust": summarize(point_trust),
        "novelty": summarize(novelty),
        "anchor_ok": summarize(anchor_ok),
        "base_ok": summarize(base_ok),
        "risky_geo_vs_current": summarize(current_cos),
        "risky_geo_vs_skeleton": summarize(skeleton_cos),
        "target_vs_current": summarize(row_cos(target, current)),
        "target_vs_skeleton": summarize(row_cos(target, skeleton.reshape(current.shape))),
        "target_vs_risky_geo": summarize(row_cos(target, risky_geo.reshape(current.shape))),
        "component_strength_mean": summarize(comp_strength.mean(axis=1)),
        "component_strength_p95": summarize(np.quantile(comp_strength, 0.95, axis=1)),
        "component_gate_frac": summarize(component_gate.mean(axis=1)),
        "component_agreement_mean": summarize(agreement.mean(axis=1)),
        "metric_ok_v118_v119": summarize(metric_ok),
        "inverse_direct_mix": summarize(direct_mix),
        "metric_template_mix": summarize(metric_mix),
        "weights_max": summarize(weights.max(axis=1)),
        "weights_entropy": summarize((-(weights * np.log(np.maximum(weights, 1e-12))).sum(axis=1) / math.log(weights.shape[1]))),
    }
    return target, weights, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--base", default=ASSETS + "/cache_v115/v115_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=ASSETS + "/cache_v120/v120_Round1_Test_Channel.npy")
    ap.add_argument("--current-target", default=ASSETS + "/cache_v115/v115_pointcloud_scene_support_target.npy")
    ap.add_argument("--v112-anchor", default=ASSETS + "/cache_v112/v112_kernel_moe_path_support_target.npy")
    ap.add_argument("--v111-anchor", default=ASSETS + "/cache_v111/v111_manual_kernel_path_support_target.npy")
    ap.add_argument("--v109-anchor", default=ASSETS + "/cache_v109/v109_accelerated_momentum_path_support_target.npy")
    ap.add_argument("--v110-risk", default=ASSETS + "/cache_v110/v110_apex_selective_momentum_path_support_target.npy")
    ap.add_argument("--v113-risk", default=ASSETS + "/cache_v113/v113_overdrive_moe_path_support_target.npy")
    ap.add_argument("--v118-risk", default=ASSETS + "/cache_v118/v118_selective_inverse_rt_apex_target.npy")
    ap.add_argument("--v119-risk", default=ASSETS + "/cache_v119/v119_hybrid_metric_collapsed_target.npy")
    ap.add_argument("--target-out", default=ASSETS + "/cache_v120/v120_consensus_component_selector_target.npy")
    ap.add_argument("--weights-out", default=ASSETS + "/cache_v120/v120_consensus_component_selector_weights.npy")
    ap.add_argument("--report", default=ASSETS + "/v120_consensus_component_selector_apex_generate.json")
    ap.add_argument("--zip-out", default=str(PATHS.round1_versions / "v120_consensus_component_selector_apex_candidate_pending.zip"))
    ap.add_argument("--candidate-zip", default=str(PATHS.round1_versions / "Round1_Submission_candidate_v120_consensus_component_selector_apex.zip"))

    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--component-vote-threshold", type=float, default=0.055)
    ap.add_argument("--skeleton-slope", type=float, default=5.5)
    ap.add_argument("--skeleton-bias", type=float, default=0.018)
    ap.add_argument("--base-top-components", type=int, default=1792)
    ap.add_argument("--proposal-top-components", type=int, default=2816)
    ap.add_argument("--novel-low", type=float, default=0.020)
    ap.add_argument("--novel-high", type=float, default=0.170)
    ap.add_argument("--anchor-low", type=float, default=0.760)
    ap.add_argument("--anchor-high", type=float, default=0.965)
    ap.add_argument("--base-low", type=float, default=0.930)
    ap.add_argument("--base-high", type=float, default=0.995)
    ap.add_argument("--agreement-low", type=float, default=0.35)
    ap.add_argument("--agreement-high", type=float, default=0.82)
    ap.add_argument("--vote-norm", type=float, default=0.62)
    ap.add_argument("--log-ratio-clip", type=float, default=1.45)
    ap.add_argument("--reg-clear", type=float, default=0.002)
    ap.add_argument("--reg-light", type=float, default=0.018)
    ap.add_argument("--reg-medium", type=float, default=1.05)
    ap.add_argument("--reg-heavy", type=float, default=1.42)
    ap.add_argument("--max-strength", type=float, default=1.05)
    ap.add_argument("--template-mix", type=float, default=0.30)
    ap.add_argument("--template-mix-cap", type=float, default=0.34)
    ap.add_argument("--metric-low", type=float, default=0.700)
    ap.add_argument("--metric-high", type=float, default=0.955)
    ap.add_argument("--inverse-direct-mix", type=float, default=0.34)
    ap.add_argument("--inverse-direct-cap", type=float, default=0.36)
    ap.add_argument("--metric-template-mix", type=float, default=0.10)
    ap.add_argument("--metric-template-cap", type=float, default=0.12)
    ap.add_argument("--target-min-cos", type=float, default=0.860)
    ap.add_argument("--weight-cols", type=int, default=14)

    # Projection is intentionally aggressive but still uses the synthetic
    # point confidence weights.
    ap.add_argument("--eta", default="clear:0.0008,light:0.006,medium:0.860,heavy:1.180")
    ap.add_argument("--min-gate", type=float, default=0.16)
    ap.add_argument("--max-gate", type=float, default=2.20)
    ap.add_argument("--conf-center", type=float, default=0.22)
    ap.add_argument("--conf-width", type=float, default=0.22)
    ap.add_argument("--entropy-low", type=float, default=0.08)
    ap.add_argument("--entropy-high", type=float, default=0.88)
    ap.add_argument("--gain-clip", default="0.34,4.20")
    ap.add_argument("--top-gain-clip", default="0.50,3.10")
    ap.add_argument("--top-components", type=int, default=3584)
    ap.add_argument("--project-batch", type=int, default=10)
    ap.add_argument("--target-floor", type=float, default=1e-12)
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    args = ap.parse_args()

    data = Path(args.data)
    pos_test = np.load(data / "Round1_Test_Pos.npy").astype(np.float32)
    geo = np.load(args.geometry_cache)
    blocked_test = geo["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked_test)

    print("load supports", flush=True)
    current = normalize_support(np.load(args.current_target, mmap_mode="r"), float(args.floor))
    v112 = normalize_support(np.load(args.v112_anchor, mmap_mode="r"), float(args.floor))
    v111 = normalize_support(np.load(args.v111_anchor, mmap_mode="r"), float(args.floor))
    v109 = normalize_support(np.load(args.v109_anchor, mmap_mode="r"), float(args.floor))
    v110 = normalize_support(np.load(args.v110_risk, mmap_mode="r"), float(args.floor))
    v113 = normalize_support(np.load(args.v113_risk, mmap_mode="r"), float(args.floor))
    v118 = normalize_support(np.load(args.v118_risk, mmap_mode="r"), float(args.floor))
    v119 = normalize_support(np.load(args.v119_risk, mmap_mode="r"), float(args.floor))

    print("build consensus component target", flush=True)
    target, weights, consensus_report = build_consensus_target(
        current,
        [current, v112, v111, v109],
        [v110, v113, v118, v119],
        regimes,
        args,
    )

    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.target_out, target.astype(np.float32))
    np.save(args.weights_out, weights.astype(np.float32))

    print("project to submission", flush=True)
    proj = projection(target, weights, regimes, pos_test, args)
    payload = {
        "method": "v120_consensus_component_selector_apex",
        "base": args.base,
        "operation": "v115 backbone; component-level consensus selector over v110/v113/v118/v119 risky targets with v115/v112/v111/v109 skeleton; phase/scalar/total power preserved",
        "online_context": {
            "best": "v115 0.497298",
            "v113_overdrive": 0.496506,
            "v116_pointcloud_family": 0.496164,
            "v119_hybrid_metric": 0.496424,
        },
        "args": vars(args),
        "consensus": consensus_report,
        "target_cos": {
            "target_vs_current": summarize(row_cos(target, current)),
            "target_vs_v112": summarize(row_cos(target, v112)),
            "target_vs_v111": summarize(row_cos(target, v111)),
            "target_vs_v109": summarize(row_cos(target, v109)),
            "target_vs_v110": summarize(row_cos(target, v110)),
            "target_vs_v113": summarize(row_cos(target, v113)),
            "target_vs_v118": summarize(row_cos(target, v118)),
            "target_vs_v119": summarize(row_cos(target, v119)),
        },
        "projection": proj,
        "outputs": {
            "submission_npy": args.output,
            "target": args.target_out,
            "weights": args.weights_out,
        },
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    def write_zip(path: str) -> None:
        files = [
            (args.output, "Round1_Test_Channel.npy"),
            (__file__, "generate_v120_consensus_component_selector_apex.py"),
            (ASSETS + "/generate_v115_pointcloud_scene_support.py", "generate_v115_pointcloud_scene_support.py"),
            (ASSETS + "/generate_v112_kernel_moe_path_support.py", "generate_v112_kernel_moe_path_support.py"),
            (ASSETS + "/generate_v98_path_ensemble_graph_attention_joint_ad.py", "generate_v98_path_ensemble_graph_attention_joint_ad.py"),
            (ASSETS + "/generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
            (ASSETS + "/powerfield_common.py", "powerfield_common.py"),
            (PATHS.code_root / "project_paths.py", "project_paths.py"),
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
            for src, arc in files:
                zf.write(src, arc)
        print(f"wrote {path}", flush=True)

    write_zip(args.zip_out)
    if args.candidate_zip:
        write_zip(args.candidate_zip)


if __name__ == "__main__":
    main()
