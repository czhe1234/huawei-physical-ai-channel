#!/usr/bin/env python3
"""Generate v118: selective inverse-RT apex support.

v117 was the first explicit train-H -> path/component attribution attempt.  Its
diagnostics show the right signal but also a serious coverage problem: official
LOS + one-bounce paths explain only about 30% of the selected train top
components.  v118 keeps the potentially valuable inverse-RT target, but uses it
as a selective high-impact expert instead of a broad replacement.

The strategy is:

* base complex channel: v115, current online best;
* conservative support backbone: v115 + v112/v111/v109 anchors;
* high-impact expert: v117 inverse-RT component attribution target;
* apply the inverse-RT expert mostly in medium/heavy regimes and only where it
  is novel but not completely inconsistent with the proven support anchors;
* preserve phase, complex scalar structure, and per-point total power.
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
from generate_v112_kernel_moe_path_support import log_blend, normalize_support, row_cos
from project_paths import PATHS
from powerfield_common import ASSETS, regime_of


REGIMES = ("clear", "light", "medium", "heavy")


def make_projection_weights(trust: np.ndarray, novelty: np.ndarray, regime: np.ndarray, cols: int = 12) -> np.ndarray:
    """Create synthetic confidence weights for projection gating.

    ``projection`` uses max weight and entropy as confidence.  The v117 raw
    path-kNN weights are high-entropy because each test path retrieves many
    attributed samples.  Here we encode the higher-level point confidence after
    checking anchor agreement and regime.
    """
    reg_boost = np.array(
        [0.03 if r == "clear" else 0.10 if r == "light" else 0.72 if r == "medium" else 1.00 for r in regime],
        dtype=np.float32,
    )
    conf = np.clip((0.60 * trust + 0.40 * novelty) * reg_boost, 0.0, 1.0).astype(np.float32)
    maxw = np.clip(0.16 + 0.68 * conf, 0.13, 0.84).astype(np.float32)
    w = np.empty((len(conf), cols), dtype=np.float32)
    w[:, 0] = maxw
    w[:, 1:] = (1.0 - maxw[:, None]) / float(cols - 1)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--base", default=ASSETS + "/cache_v115/v115_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=ASSETS + "/cache_v118/v118_Round1_Test_Channel.npy")
    ap.add_argument("--current-target", default=ASSETS + "/cache_v115/v115_pointcloud_scene_support_target.npy")
    ap.add_argument("--inverse-target", default=ASSETS + "/cache_v117/v117_inverse_rt_component_target.npy")
    ap.add_argument("--inverse-weights", default=ASSETS + "/cache_v117/v117_inverse_rt_component_weights.npy")
    ap.add_argument("--v112-anchor", default=ASSETS + "/cache_v112/v112_kernel_moe_path_support_target.npy")
    ap.add_argument("--v111-anchor", default=ASSETS + "/cache_v111/v111_manual_kernel_path_support_target.npy")
    ap.add_argument("--v109-anchor", default=ASSETS + "/cache_v109/v109_accelerated_momentum_path_support_target.npy")
    ap.add_argument("--target-out", default=ASSETS + "/cache_v118/v118_selective_inverse_rt_apex_target.npy")
    ap.add_argument("--weights-out", default=ASSETS + "/cache_v118/v118_selective_inverse_rt_apex_weights.npy")
    ap.add_argument("--report", default=ASSETS + "/v118_selective_inverse_rt_apex_generate.json")
    ap.add_argument("--zip-out", default=str(PATHS.round1_versions / "v118_selective_inverse_rt_apex_candidate_pending.zip"))
    ap.add_argument("--candidate-zip", default=str(PATHS.round1_versions / "Round1_Submission_candidate_v118_selective_inverse_rt_apex.zip"))

    # Strength of the inverse-RT expert before projection confidence gating.
    ap.add_argument("--base-strength-clear", type=float, default=0.000)
    ap.add_argument("--base-strength-light", type=float, default=0.020)
    ap.add_argument("--base-strength-medium", type=float, default=0.560)
    ap.add_argument("--base-strength-heavy", type=float, default=0.780)
    ap.add_argument("--novel-low", type=float, default=0.055)
    ap.add_argument("--novel-high", type=float, default=0.500)
    ap.add_argument("--agree-low", type=float, default=0.330)
    ap.add_argument("--agree-high", type=float, default=0.820)
    ap.add_argument("--anchor-low", type=float, default=0.300)
    ap.add_argument("--anchor-high", type=float, default=0.780)
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--batch", type=int, default=12)

    # Projection parameters.  Eta is intentionally higher than v117 for trusted
    # medium/heavy points, but clear/light still stay near v115.
    ap.add_argument("--eta", default="clear:0.0005,light:0.006,medium:0.980,heavy:1.320")
    ap.add_argument("--min-gate", type=float, default=0.18)
    ap.add_argument("--max-gate", type=float, default=2.35)
    ap.add_argument("--conf-center", type=float, default=0.20)
    ap.add_argument("--conf-width", type=float, default=0.22)
    ap.add_argument("--entropy-low", type=float, default=0.10)
    ap.add_argument("--entropy-high", type=float, default=0.86)
    ap.add_argument("--gain-clip", default="0.32,4.25")
    ap.add_argument("--top-gain-clip", default="0.48,3.10")
    ap.add_argument("--top-components", type=int, default=2816)
    ap.add_argument("--project-batch", type=int, default=10)
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    args = ap.parse_args()
    args.target_floor = float(args.floor)

    data = Path(args.data)
    pos_test = np.load(data / "Round1_Test_Pos.npy").astype(np.float32)
    geo = np.load(args.geometry_cache)
    blocked_test = geo["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked_test)

    current = normalize_support(np.load(args.current_target), float(args.floor))
    inverse = normalize_support(np.load(args.inverse_target), float(args.floor))
    v112 = normalize_support(np.load(args.v112_anchor), float(args.floor))
    v111 = normalize_support(np.load(args.v111_anchor), float(args.floor))
    v109 = normalize_support(np.load(args.v109_anchor), float(args.floor))
    inv_w_raw = np.load(args.inverse_weights).astype(np.float32)

    cos_inv_cur = row_cos(inverse, current).astype(np.float32)
    cos_inv_v112 = row_cos(inverse, v112).astype(np.float32)
    cos_inv_v111 = row_cos(inverse, v111).astype(np.float32)
    cos_inv_v109 = row_cos(inverse, v109).astype(np.float32)
    anchor_agree = np.maximum.reduce([cos_inv_v112, cos_inv_v111, cos_inv_v109]).astype(np.float32)
    novelty = np.clip((1.0 - cos_inv_cur - float(args.novel_low)) / max(float(args.novel_high) - float(args.novel_low), 1e-6), 0.0, 1.0)
    not_too_far = np.clip((cos_inv_cur - float(args.agree_low)) / max(float(args.agree_high) - float(args.agree_low), 1e-6), 0.0, 1.0)
    anchor_ok = np.clip((anchor_agree - float(args.anchor_low)) / max(float(args.anchor_high) - float(args.anchor_low), 1e-6), 0.0, 1.0)
    raw_max = inv_w_raw.max(axis=1)
    raw_entropy = -(inv_w_raw * np.log(np.maximum(inv_w_raw, 1e-12))).sum(axis=1) / math.log(inv_w_raw.shape[1])
    inv_conf = np.clip((raw_max - 0.035) / 0.085, 0.0, 1.0) * np.clip((0.975 - raw_entropy) / 0.12, 0.0, 1.0)
    trust = np.clip(0.40 * not_too_far + 0.35 * anchor_ok + 0.25 * inv_conf, 0.0, 1.0).astype(np.float32)

    base_strength = np.array(
        [
            float(args.base_strength_clear)
            if r == "clear"
            else float(args.base_strength_light)
            if r == "light"
            else float(args.base_strength_medium)
            if r == "medium"
            else float(args.base_strength_heavy)
            for r in regimes
        ],
        dtype=np.float32,
    )
    # Sweet spot: use inverse-RT where it is new enough to matter but still
    # anchored by proven supports.  Very low-current agreement gets damped.
    strength = base_strength * (0.25 + 0.75 * trust) * (0.40 + 0.60 * novelty)
    strength *= np.clip((cos_inv_cur - 0.18) / 0.30, 0.0, 1.0)
    strength = np.clip(strength, 0.0, 0.72).astype(np.float32)

    coeff = np.zeros((len(current), 5), dtype=np.float32)
    coeff[:, 0] = 1.0 - strength
    coeff[:, 1] = 0.920 * strength
    coeff[:, 2] = 0.050 * strength
    coeff[:, 3] = 0.020 * strength
    coeff[:, 4] = 0.010 * strength
    target = log_blend([current, inverse, v112, v111, v109], coeff, float(args.floor), int(args.batch))
    point_w = make_projection_weights(trust, novelty, regimes, cols=12)

    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.target_out, target.astype(np.float32))
    np.save(args.weights_out, point_w.astype(np.float32))

    proj = projection(target, point_w, regimes, pos_test, args)
    payload = {
        "method": "v118_selective_inverse_rt_apex",
        "base": args.base,
        "operation": "v115 support backbone plus selective v117 inverse-RT component-attribution expert; phase/scalar/total power preserved",
        "online_context": {
            "best": "v115 0.497298",
            "v116": 0.496164,
            "v117": "pending at generation time",
        },
        "args": vars(args),
        "selection": {
            "cos_inverse_current": summarize(cos_inv_cur),
            "anchor_agreement": summarize(anchor_agree),
            "novelty": summarize(novelty),
            "not_too_far": summarize(not_too_far),
            "anchor_ok": summarize(anchor_ok),
            "inverse_raw_max": summarize(raw_max),
            "inverse_raw_entropy": summarize(raw_entropy),
            "inverse_conf": summarize(inv_conf),
            "trust": summarize(trust),
            "strength": summarize(strength),
        },
        "target_cos": {
            "target_vs_current": summarize(row_cos(target, current)),
            "target_vs_inverse": summarize(row_cos(target, inverse)),
            "target_vs_v112": summarize(row_cos(target, v112)),
            "target_vs_v111": summarize(row_cos(target, v111)),
            "target_vs_v109": summarize(row_cos(target, v109)),
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
            (__file__, "generate_v118_selective_inverse_rt_apex.py"),
            (ASSETS + "/generate_v117_inverse_rt_component_attribution.py", "generate_v117_inverse_rt_component_attribution.py"),
            (ASSETS + "/generate_v115_pointcloud_scene_support.py", "generate_v115_pointcloud_scene_support.py"),
            (ASSETS + "/generate_v112_kernel_moe_path_support.py", "generate_v112_kernel_moe_path_support.py"),
            (ASSETS + "/generate_v98_path_ensemble_graph_attention_joint_ad.py", "generate_v98_path_ensemble_graph_attention_joint_ad.py"),
            (ASSETS + "/generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
            (ASSETS + "/powerfield_common.py", "powerfield_common.py"),
            (PATHS.code_root / "project_paths.py", "project_paths.py"),
            (ASSETS + "/build_official_pointcloud_scene.py", "build_official_pointcloud_scene.py"),
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
