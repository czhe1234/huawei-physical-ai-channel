#!/usr/bin/env python3
"""Build a v125-equivalent cross-fit train AD backbone.

v126 learned allocation residuals from a v98-like LOO base but applied them to
the much later v125 test backbone.  This script removes that distribution
mismatch.  For every train query, all geometry/visibility neighbors exclude the
query itself.  It then applies a v124-like collapsed-support projection and the
v125 template-conditioned pol/UE allocation projection to the v98-like LOO AD
base, while keeping the original train truth as base + residual.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import MH, MP, MV, N_ANT, S_SC, summarize
from generate_v112_kernel_moe_path_support import (
    entropy01,
    log_average,
    log_blend,
    normalize_support,
    pairwise_knn,
    row_cos,
    softmax_np,
    standardize,
    weights_from_dist,
)
from generate_v115_pointcloud_scene_support import official_scene_embedding, spatial_embedding
from generate_v121_support_template_manifold import build_prototypes, support_signature, template_blend, topk_prob
from generate_v124_visibility_misleading_overdrive import probability_from_labels, visibility_features
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


REGIMES = ("clear", "light", "medium", "heavy")


def parse_regime_values(text: str) -> dict[str, float]:
    out = {r: 0.0 for r in REGIMES}
    for item in text.split(","):
        key, value = item.split(":")
        out[key.strip()] = float(value)
    return out


def blend_allocation_batch(
    allocations: np.ndarray,
    probabilities: np.ndarray,
    top: int,
    floor: float,
) -> np.ndarray:
    k = min(int(top), probabilities.shape[1])
    idx = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    weights = np.take_along_axis(probabilities, idx, axis=1).astype(np.float64)
    order = np.argsort(-weights, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    weights = np.take_along_axis(weights, order, axis=1)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    acc = np.zeros((len(probabilities), MP, MH, MV, N_ANT, S_SC), dtype=np.float64)
    for j in range(k):
        acc += weights[:, j, None, None, None, None, None] * np.log(
            np.maximum(allocations[idx[:, j]], floor)
        )
    out = np.exp(acc).astype(np.float32)
    out /= np.maximum(out.sum(axis=(1, 4), keepdims=True), floor)
    return out


def top_component_mask(collapsed: np.ndarray, k: int) -> np.ndarray:
    flat = collapsed.reshape(len(collapsed), -1)
    kk = min(int(k), flat.shape[1])
    idx = np.argpartition(flat, -kk, axis=1)[:, -kk:]
    mask = np.zeros_like(flat, dtype=bool)
    mask[np.arange(len(flat))[:, None], idx] = True
    return mask.reshape(len(flat), MH, MV, S_SC)


def apply_support_projection(
    ad: np.ndarray,
    target: np.ndarray,
    eta: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    p0 = (ad.real**2 + ad.imag**2).astype(np.float32)
    coll0 = p0.sum(axis=(1, 4))
    total0 = np.maximum(coll0.sum(axis=(1, 2, 3)), TINY)
    current = coll0 / total0[:, None, None, None]
    ratio = ((target + float(args.floor)) / (current + float(args.floor))) ** eta[:, None, None, None]
    gain = np.sqrt(ratio).astype(np.float32)
    gain = np.clip(gain, float(args.support_gain_low), float(args.support_gain_high))
    mask = top_component_mask(coll0, int(args.support_top_components))
    gain = np.where(
        mask,
        np.clip(gain, float(args.support_top_gain_low), float(args.support_top_gain_high)),
        gain,
    )
    out = (ad * gain[:, None, :, :, None, :]).astype(np.complex64)
    total1 = np.maximum(np.sum(np.abs(out) ** 2, axis=(1, 2, 3, 4, 5)), TINY)
    out *= np.sqrt(total0 / total1)[:, None, None, None, None, None].astype(np.float32)
    return out


def apply_allocation_projection(
    ad: np.ndarray,
    target: np.ndarray,
    eta: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    p0 = (ad.real**2 + ad.imag**2).astype(np.float32)
    coll0 = p0.sum(axis=(1, 4))
    current = p0 / np.maximum(coll0[:, None, :, :, None, :], float(args.alloc_floor))
    ratio = ((target + float(args.ratio_floor)) / (current + float(args.ratio_floor))) ** eta[
        :, None, None, None, None, None
    ]
    gain = np.sqrt(ratio).astype(np.float32)
    gain = np.clip(gain, float(args.alloc_gain_low), float(args.alloc_gain_high))
    mask = top_component_mask(coll0, int(args.alloc_top_components))[:, None, :, :, None, :]
    gain = np.where(
        mask,
        np.clip(gain, float(args.alloc_top_gain_low), float(args.alloc_top_gain_high)),
        1.0,
    )
    out = (ad * gain).astype(np.complex64)
    coll1 = np.maximum(np.sum(np.abs(out) ** 2, axis=(1, 4)), TINY)
    out *= np.sqrt(coll0 / coll1)[:, None, :, :, None, :].astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--scene-cache", default=ASSETS + "/official_pointcloud_scene_cache.npz")
    ap.add_argument("--truth-support", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--train-current-support", default=ASSETS + "/train_v98like_loo_joint_target.npy")
    ap.add_argument("--labels", default=ASSETS + "/cache_v121/v121_support_template_manifold_train_labels.npy")
    ap.add_argument("--allocations", default=ASSETS + "/cache_v125/v125_template_polue_allocations.npy")
    ap.add_argument("--input-base-ad", default=ASSETS + "/train_v98like_loo_base_ad.npy")
    ap.add_argument("--input-residual-ad", default=ASSETS + "/train_v98like_loo_residual_ad.npy")
    ap.add_argument("--output-base-ad", default=ASSETS + "/train_v125eq_crossfit_base_ad.npy")
    ap.add_argument("--output-residual-ad", default=ASSETS + "/train_v125eq_crossfit_residual_ad.npy")
    ap.add_argument("--prob-out", default=ASSETS + "/train_v125eq_crossfit_template_probs.npy")
    ap.add_argument("--support-target-out", default=ASSETS + "/train_v125eq_crossfit_support_target.npy")
    ap.add_argument("--report", default=ASSETS + "/train_v125eq_crossfit_report.json")
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--target-batch", type=int, default=16)
    ap.add_argument("--k-scene", type=int, default=18)
    ap.add_argument("--k-geom", type=int, default=28)
    ap.add_argument("--k-vis", type=int, default=26)
    ap.add_argument("--scene-power", type=float, default=3.05)
    ap.add_argument("--geom-power", type=float, default=2.60)
    ap.add_argument("--vis-power", type=float, default=3.55)
    ap.add_argument("--scene-floor", type=float, default=0.50)
    ap.add_argument("--geom-floor", type=float, default=0.62)
    ap.add_argument("--vis-floor", type=float, default=0.45)
    ap.add_argument("--proto-temp", type=float, default=0.020)
    ap.add_argument("--scene-weight", type=float, default=0.34)
    ap.add_argument("--geom-weight", type=float, default=0.16)
    ap.add_argument("--vis-weight", type=float, default=0.30)
    ap.add_argument("--current-weight", type=float, default=0.20)
    ap.add_argument("--support-eta", default="clear:0.00015,light:0.0025,medium:0.720,heavy:1.020")
    ap.add_argument("--support-min-gate", type=float, default=0.20)
    ap.add_argument("--support-max-gate", type=float, default=2.55)
    ap.add_argument("--support-conf-center", type=float, default=0.17)
    ap.add_argument("--support-conf-width", type=float, default=0.15)
    ap.add_argument("--support-entropy-low", type=float, default=0.20)
    ap.add_argument("--support-entropy-high", type=float, default=0.92)
    ap.add_argument("--support-gain-low", type=float, default=0.36)
    ap.add_argument("--support-gain-high", type=float, default=3.75)
    ap.add_argument("--support-top-gain-low", type=float, default=0.50)
    ap.add_argument("--support-top-gain-high", type=float, default=2.75)
    ap.add_argument("--support-top-components", type=int, default=2816)
    ap.add_argument("--allocation-eta", default="clear:0.004,light:0.018,medium:0.260,heavy:0.360")
    ap.add_argument("--allocation-min-gate", type=float, default=0.45)
    ap.add_argument("--allocation-max-gate", type=float, default=1.35)
    ap.add_argument("--allocation-conf-center", type=float, default=0.18)
    ap.add_argument("--allocation-conf-width", type=float, default=0.34)
    ap.add_argument("--allocation-entropy-low", type=float, default=0.18)
    ap.add_argument("--allocation-entropy-high", type=float, default=0.96)
    ap.add_argument("--alloc-gain-low", type=float, default=0.62)
    ap.add_argument("--alloc-gain-high", type=float, default=1.62)
    ap.add_argument("--alloc-top-gain-low", type=float, default=0.58)
    ap.add_argument("--alloc-top-gain-high", type=float, default=1.75)
    ap.add_argument("--alloc-top-components", type=int, default=1536)
    ap.add_argument("--alloc-floor", type=float, default=1e-10)
    ap.add_argument("--ratio-floor", type=float, default=2e-5)
    ap.add_argument("--occlusion-samples", type=int, default=120)
    ap.add_argument("--occlusion-batch", type=int, default=4096)
    ap.add_argument("--vertical-threshold", type=int, default=2)
    ap.add_argument("--z-margin", type=float, default=0.35)
    ap.add_argument("--min-reflection-score", type=float, default=1e-5)
    args = ap.parse_args()
    args.target_floor = float(args.floor)

    data = Path(args.data)
    pos = np.load(data / "Round1_Train_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked = geo["train_blocked"].astype(np.float64)
    regimes = regime_of(blocked)
    scene = np.load(args.scene_cache, allow_pickle=True)
    truth_support = normalize_support(np.load(args.truth_support, mmap_mode="r"), float(args.floor))
    current = normalize_support(np.load(args.train_current_support, mmap_mode="r"), float(args.floor))
    labels = np.load(args.labels).astype(np.int64)
    clusters = int(labels.max()) + 1
    prototypes = build_prototypes(truth_support, labels, clusters, float(args.floor))

    print("build cross-fit v124 feature banks", flush=True)
    walls = scene["wall_patches"].astype(np.float32)
    cloud_raw = official_scene_embedding(
        pos, blocked, scene["train_features"].astype(np.float32),
        scene["train_reflection_score"].astype(np.float32),
        scene["train_reflection_path_length"].astype(np.float32),
        scene["train_reflection_excess_length"].astype(np.float32),
        scene["train_reflection_point"].astype(np.float32),
        scene["train_reflection_patch_id"].astype(np.int64), walls,
    )
    vis_raw, vis_info = visibility_features(pos, blocked, scene, "train", args)
    sig_raw = support_signature(current, float(args.floor))
    spatial_raw = spatial_embedding(pos, blocked)
    cloud, _ = standardize(cloud_raw, cloud_raw)
    vis, _ = standardize(vis_raw, vis_raw)
    sig, _ = standardize(sig_raw, sig_raw)
    spatial, _ = standardize(spatial_raw, spatial_raw)
    scene_bank = np.concatenate([cloud, 0.52 * sig, 0.95 * vis], axis=1).astype(np.float32)
    geom_bank = np.concatenate([spatial, 0.78 * sig, 0.35 * vis], axis=1).astype(np.float32)
    visibility_bank = np.concatenate([1.55 * vis, 0.38 * sig, 0.18 * spatial], axis=1).astype(np.float32)

    idx_scene, d_scene = pairwise_knn(scene_bank, scene_bank, int(args.k_scene), exclude_self=True)
    idx_geom, d_geom = pairwise_knn(geom_bank, geom_bank, int(args.k_geom), exclude_self=True)
    idx_vis, d_vis = pairwise_knn(visibility_bank, visibility_bank, int(args.k_vis), exclude_self=True)
    w_scene = weights_from_dist(d_scene, float(args.scene_power), float(args.scene_floor))
    w_geom = weights_from_dist(d_geom, float(args.geom_power), float(args.geom_floor))
    w_vis = weights_from_dist(d_vis, float(args.vis_power), float(args.vis_floor))
    prob_scene = probability_from_labels(idx_scene, w_scene, labels, clusters)
    prob_geom = probability_from_labels(idx_geom, w_geom, labels, clusters)
    prob_vis = probability_from_labels(idx_vis, w_vis, labels, clusters)

    proto_flat = prototypes.reshape(clusters, -1).astype(np.float32)
    proto_norm = proto_flat / np.maximum(np.linalg.norm(proto_flat, axis=1, keepdims=True), TINY)
    cur_flat = current.reshape(len(current), -1).astype(np.float32)
    cur_norm = cur_flat / np.maximum(np.linalg.norm(cur_flat, axis=1, keepdims=True), TINY)
    prob_cur = softmax_np(cur_norm @ proto_norm.T, temp=float(args.proto_temp))
    sw, gw, vw, cw = (float(args.scene_weight), float(args.geom_weight), float(args.vis_weight), float(args.current_weight))
    probabilities = (sw * prob_scene + gw * prob_geom + vw * prob_vis + cw * prob_cur) / (sw + gw + vw + cw)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), TINY)
    np.save(args.prob_out, probabilities.astype(np.float32))

    print("build cross-fit support target", flush=True)
    template_target = template_blend(prototypes, probabilities, float(args.floor), int(args.target_batch))
    scene_anchor = log_average(truth_support, idx_scene, w_scene, float(args.floor), int(args.target_batch))
    geom_anchor = log_average(truth_support, idx_geom, w_geom, float(args.floor), int(args.target_batch))
    vis_anchor = log_average(truth_support, idx_vis, w_vis, float(args.floor), int(args.target_batch))
    anchor_coeff = np.tile(np.asarray([[0.40, 0.30, 0.30]], dtype=np.float32), (len(pos), 1))
    local_anchor = log_blend([scene_anchor, geom_anchor, vis_anchor], anchor_coeff, float(args.floor), int(args.target_batch))

    scene_vis_agree = np.sum(np.sqrt(np.maximum(prob_scene, 0.0) * np.maximum(prob_vis, 0.0)), axis=1)
    vis_cur_agree = np.sum(np.sqrt(np.maximum(prob_vis, 0.0) * np.maximum(prob_cur, 0.0)), axis=1)
    template_cos = row_cos(template_target, current).astype(np.float32)
    prob_top = probabilities.max(axis=1).astype(np.float32)
    aux = vis_info["aux"]
    visible = np.clip(aux["top_clear_score"] / 0.55, 0.0, 1.0)
    misleading = np.clip(aux["misleading_reflection"] / 0.55, 0.0, 1.0)
    no_path = aux["no_visible_path"].astype(np.float32)
    divergence = np.clip((1.0 - template_cos - 0.030) / 0.115, 0.0, 1.0)
    overdrive = (
        np.clip((misleading - 0.42) / 0.48, 0.0, 1.0)
        * (0.42 + 0.58 * divergence)
        * (0.62 + 0.38 * np.clip((prob_top - 0.18) / 0.35, 0.0, 1.0))
    )
    reg_strength = {"clear": 0.004, "light": 0.020, "medium": 0.660, "heavy": 0.930}
    strength = np.asarray([reg_strength[str(r)] for r in regimes], dtype=np.float32)
    strength *= 0.48 + 0.52 * np.clip((prob_top - 0.12) / 0.28, 0.0, 1.0)
    strength *= 0.54 + 0.46 * np.clip((scene_vis_agree - 0.18) / 0.36, 0.0, 1.0)
    strength *= 0.58 + 0.42 * np.clip((vis_cur_agree - 0.22) / 0.36, 0.0, 1.0)
    strength *= 0.58 + 0.42 * np.clip((template_cos - 0.84) / 0.12, 0.0, 1.0)
    strength *= 0.80 + 0.22 * misleading + 0.10 * visible + 0.36 * overdrive
    strength *= 1.0 - 0.16 * no_path * (1.0 - misleading)
    strength = np.clip(strength, 0.0, 1.02).astype(np.float32)
    coeff = np.column_stack([1.0 - strength, 0.91 * strength, 0.09 * strength]).astype(np.float32)
    support_target = log_blend(
        [current, template_target, local_anchor], coeff, float(args.floor), int(args.target_batch)
    )
    np.save(args.support_target_out, support_target.astype(np.float32))

    top_prob = topk_prob(probabilities, 12)
    maxw = top_prob.max(axis=1)
    ent = entropy01(top_prob)
    support_conf = np.clip((maxw - float(args.support_conf_center)) / float(args.support_conf_width), 0.0, 1.0)
    support_conf *= np.clip(
        (float(args.support_entropy_high) - ent) /
        (float(args.support_entropy_high) - float(args.support_entropy_low)), 0.0, 1.0
    )
    support_eta_base = parse_regime_values(args.support_eta)
    support_eta = np.asarray([support_eta_base[str(r)] for r in regimes], dtype=np.float32)
    support_eta *= float(args.support_min_gate) + (
        float(args.support_max_gate) - float(args.support_min_gate)
    ) * support_conf.astype(np.float32)

    prob_ent = entropy01(probabilities)
    alloc_conf = np.clip(
        (prob_top - float(args.allocation_conf_center)) / float(args.allocation_conf_width), 0.0, 1.0
    )
    alloc_conf *= np.clip(
        (float(args.allocation_entropy_high) - prob_ent) /
        (float(args.allocation_entropy_high) - float(args.allocation_entropy_low)), 0.0, 1.0
    )
    alloc_eta_base = parse_regime_values(args.allocation_eta)
    alloc_eta = np.asarray([alloc_eta_base[str(r)] for r in regimes], dtype=np.float32)
    alloc_eta *= float(args.allocation_min_gate) + (
        float(args.allocation_max_gate) - float(args.allocation_min_gate)
    ) * alloc_conf.astype(np.float32)

    print("project v125-equivalent train AD", flush=True)
    old_base = np.load(args.input_base_ad, mmap_mode="r")
    old_residual = np.load(args.input_residual_ad, mmap_mode="r")
    allocations = np.load(args.allocations, mmap_mode="r")
    out_base = np.lib.format.open_memmap(
        args.output_base_ad, mode="w+", dtype=np.complex64, shape=old_base.shape
    )
    out_residual = np.lib.format.open_memmap(
        args.output_residual_ad, mode="w+", dtype=np.complex64, shape=old_base.shape
    )
    delta_num = delta_den = 0.0
    truth_num = truth_den = 0.0
    for start in range(0, len(pos), int(args.batch)):
        end = min(start + int(args.batch), len(pos))
        base = np.array(old_base[start:end], dtype=np.complex64, copy=True)
        truth = base + np.asarray(old_residual[start:end], dtype=np.complex64)
        stage = apply_support_projection(base, support_target[start:end], support_eta[start:end], args)
        alloc_target = blend_allocation_batch(
            allocations, probabilities[start:end], 12, float(args.alloc_floor)
        )
        stage = apply_allocation_projection(stage, alloc_target, alloc_eta[start:end], args)
        out_base[start:end] = stage
        out_residual[start:end] = truth - stage
        out_base.flush()
        out_residual.flush()
        delta_num += float(np.sum(np.abs(stage - base) ** 2, dtype=np.float64))
        delta_den += float(np.sum(np.abs(base) ** 2, dtype=np.float64))
        truth_num += float(np.sum(np.abs(stage - truth) ** 2, dtype=np.float64))
        truth_den += float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
        print(f"v125eq train base {end}/{len(pos)}", flush=True)

    payload = {
        "method": "v125_equivalent_crossfit_train_base",
        "note": "train-query KNN branches exclude self; template prototypes/statistics use aggregate train truth",
        "support_target_vs_current": summarize(row_cos(support_target, current)),
        "support_strength": summarize(strength),
        "support_eta": summarize(support_eta),
        "allocation_eta": summarize(alloc_eta),
        "probability_top1": summarize(prob_top),
        "probability_entropy": summarize(prob_ent),
        "delta_nmse_to_v98like_base": float(delta_num / max(delta_den, TINY)),
        "nmse_to_truth": float(truth_num / max(truth_den, TINY)),
        "outputs": {
            "base_ad": args.output_base_ad,
            "residual_ad": args.output_residual_ad,
            "probabilities": args.prob_out,
            "support_target": args.support_target_out,
        },
        "args": vars(args),
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
