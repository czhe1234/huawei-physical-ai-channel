#!/usr/bin/env python3
"""Generate v128: bad-point support residual templates.

This version stops refining the 2-pol x 4-UE allocation.  It mines the
remaining support errors directly from training truth:

    true collapsed AD support - v125-equivalent cross-fit support

The high-error training points are clustered in residual-ratio space to form
"missing propagation-mode" patches.  Test points select those patches using
official point-cloud geometry, current v125 support shape, and v124 template
probabilities.  The final projection changes only the collapsed H/V/delay
support.  It preserves v125 phase, per-point total power, and v125 pol/UE
allocation structure.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import MH, MV, S_SC, summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from generate_v112_kernel_moe_path_support import (
    entropy01,
    log_blend,
    normalize_support,
    pairwise_knn,
    row_cos,
    softmax_np,
    standardize,
    weights_from_dist,
)
from generate_v115_pointcloud_scene_support import official_scene_embedding, spatial_embedding
from generate_v121_support_template_manifold import (
    cluster_embeddings,
    fit_support_svd,
    support_signature,
    topk_prob,
)
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


REGIMES = ("clear", "light", "medium", "heavy")


def parse_regime_values(text: str, default: float) -> dict[str, float]:
    out = {r: float(default) for r in REGIMES}
    if not text:
        return out
    for item in text.split(","):
        if not item:
            continue
        key, value = item.split(":")
        key = key.strip()
        if key not in out:
            raise ValueError(f"unknown regime {key!r}")
        out[key] = float(value)
    return out


def collapsed_support_from_ad(
    ad_path: str,
    out_path: str,
    floor: float,
    batch: int,
) -> np.ndarray:
    out = Path(out_path)
    if out.exists():
        return normalize_support(np.load(out, mmap_mode="r"), floor)

    ad = np.load(ad_path, mmap_mode="r")
    supp = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(ad.shape[0], MH, MV, S_SC))
    for start in range(0, ad.shape[0], int(batch)):
        end = min(start + int(batch), ad.shape[0])
        block = np.asarray(ad[start:end], dtype=np.complex64)
        p = (block.real**2 + block.imag**2).astype(np.float32)
        coll = p.sum(axis=(1, 4))
        coll /= np.maximum(coll.sum(axis=(1, 2, 3), keepdims=True), TINY)
        if floor > 0:
            coll = np.maximum(coll, floor).astype(np.float32)
            coll /= np.maximum(coll.sum(axis=(1, 2, 3), keepdims=True), TINY)
        supp[start:end] = coll.astype(np.float32)
        supp.flush()
        print(f"train base support {end}/{ad.shape[0]}", flush=True)
    return normalize_support(np.load(out_path, mmap_mode="r"), floor)


def build_bad_mask(
    train_base: np.ndarray,
    truth: np.ndarray,
    regimes: np.ndarray,
    bad_quantile: float,
    min_bad_cos: float,
) -> tuple[np.ndarray, np.ndarray]:
    cos = row_cos(train_base, truth).astype(np.float32)
    badness = (1.0 - cos).astype(np.float32)
    threshold = max(float(np.quantile(badness, bad_quantile)), 1.0 - float(min_bad_cos))
    mask = badness >= threshold
    # Keep more medium/heavy cases because those are where the existing
    # support-template stack still misses non-LOS propagation modes.
    mask |= ((regimes == "medium") | (regimes == "heavy")) & (badness >= 0.55 * threshold)
    # Avoid tiny clusters if the threshold is too aggressive.
    if int(mask.sum()) < 256:
        ids = np.argsort(badness)[-256:]
        mask[ids] = True
    return mask.astype(bool), cos


def residual_ratio_templates(
    train_base: np.ndarray,
    truth: np.ndarray,
    bad_mask: np.ndarray,
    clusters: int,
    svd_components: int,
    seed: int,
    floor: float,
    ratio_clip: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    bad_base = train_base[bad_mask]
    bad_truth = truth[bad_mask]
    log_ratio = np.log(np.maximum(bad_truth, floor)) - np.log(np.maximum(bad_base, floor))
    log_ratio = np.clip(log_ratio, math.log(ratio_clip[0]), math.log(ratio_clip[1])).astype(np.float32)
    flat = log_ratio.reshape(log_ratio.shape[0], -1)
    emb, _, evr = fit_support_svd(flat, int(svd_components), int(seed))
    labels = cluster_embeddings(emb, int(clusters), int(seed))
    k = int(labels.max()) + 1

    proto_ratio = np.empty((k, MH, MV, S_SC), dtype=np.float32)
    proto_truth = np.empty((k, MH, MV, S_SC), dtype=np.float32)
    global_lr = log_ratio.mean(axis=0)
    global_truth_log = np.log(np.maximum(bad_truth, floor)).mean(axis=0)
    for j in range(k):
        idx = labels == j
        if np.any(idx):
            lr = log_ratio[idx].mean(axis=0)
            tl = np.log(np.maximum(bad_truth[idx], floor)).mean(axis=0)
        else:
            lr = global_lr
            tl = global_truth_log
        ratio = np.exp(np.clip(lr, math.log(ratio_clip[0]), math.log(ratio_clip[1]))).astype(np.float32)
        proto_ratio[j] = ratio
        p = np.exp(tl).astype(np.float32)
        p /= np.maximum(p.sum(), TINY)
        proto_truth[j] = p

    counts = np.bincount(labels, minlength=k)
    report = {
        "bad_count": int(bad_mask.sum()),
        "clusters": int(k),
        "cluster_count": summarize(counts),
        "svd_explained_variance_sum": float(np.sum(evr)),
        "ratio_clip": list(ratio_clip),
    }
    return proto_ratio, proto_truth, labels.astype(np.int64), report


def mixture_ratio(
    proto_ratio: np.ndarray,
    prob: np.ndarray,
    floor: float,
    batch: int,
) -> np.ndarray:
    out = np.empty((prob.shape[0], MH, MV, S_SC), dtype=np.float32)
    log_ratio = np.log(np.maximum(proto_ratio, floor)).astype(np.float32)
    for start in range(0, prob.shape[0], int(batch)):
        end = min(start + int(batch), prob.shape[0])
        lr = np.tensordot(prob[start:end].astype(np.float32), log_ratio, axes=(1, 0)).astype(np.float32)
        out[start:end] = np.exp(lr).astype(np.float32)
        print(f"ratio mix {end}/{prob.shape[0]}", flush=True)
    return out


def apply_ratio_support(current: np.ndarray, ratio: np.ndarray, floor: float) -> np.ndarray:
    target = current.astype(np.float32) * ratio.astype(np.float32)
    target = np.maximum(target, floor).astype(np.float32)
    target /= np.maximum(target.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return target.astype(np.float32)


def probability_from_bad_neighbors(
    idx: np.ndarray,
    weights: np.ndarray,
    bad_cluster_labels: np.ndarray,
    cluster_count: int,
) -> np.ndarray:
    eye = np.eye(cluster_count, dtype=np.float32)
    prob = (weights[:, :, None] * eye[bad_cluster_labels[idx]]).sum(axis=1).astype(np.float32)
    prob /= np.maximum(prob.sum(axis=1, keepdims=True), TINY)
    return prob


def build_features(
    pos_train: np.ndarray,
    pos_test: np.ndarray,
    blocked_train: np.ndarray,
    blocked_test: np.ndarray,
    train_support: np.ndarray,
    test_support: np.ndarray,
    train_template_prob: np.ndarray,
    test_template_prob: np.ndarray,
    scene: np.lib.npyio.NpzFile,
    floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    wall_patches = scene["wall_patches"].astype(np.float32)
    cloud_train_raw = official_scene_embedding(
        pos_train,
        blocked_train,
        scene["train_features"].astype(np.float32),
        scene["train_reflection_score"].astype(np.float32),
        scene["train_reflection_path_length"].astype(np.float32),
        scene["train_reflection_excess_length"].astype(np.float32),
        scene["train_reflection_point"].astype(np.float32),
        scene["train_reflection_patch_id"].astype(np.int64),
        wall_patches,
    )
    cloud_test_raw = official_scene_embedding(
        pos_test,
        blocked_test,
        scene["test_features"].astype(np.float32),
        scene["test_reflection_score"].astype(np.float32),
        scene["test_reflection_path_length"].astype(np.float32),
        scene["test_reflection_excess_length"].astype(np.float32),
        scene["test_reflection_point"].astype(np.float32),
        scene["test_reflection_patch_id"].astype(np.int64),
        wall_patches,
    )
    spatial_train_raw = spatial_embedding(pos_train, blocked_train)
    spatial_test_raw = spatial_embedding(pos_test, blocked_test)
    sig_train_raw = support_signature(train_support, floor)
    sig_test_raw = support_signature(test_support, floor)

    cloud_train, cloud_test = standardize(cloud_train_raw, cloud_test_raw)
    spatial_train, spatial_test = standardize(spatial_train_raw, spatial_test_raw)
    sig_train, sig_test = standardize(sig_train_raw, sig_test_raw)
    prob_train, prob_test = standardize(train_template_prob.astype(np.float32), test_template_prob.astype(np.float32))

    feat_train = np.concatenate(
        [0.82 * cloud_train, 0.50 * spatial_train, 0.72 * sig_train, 0.42 * prob_train],
        axis=1,
    ).astype(np.float32)
    feat_test = np.concatenate(
        [0.82 * cloud_test, 0.50 * spatial_test, 0.72 * sig_test, 0.42 * prob_test],
        axis=1,
    ).astype(np.float32)
    report = {
        "feature_dim": int(feat_train.shape[1]),
        "cloud_dim": int(cloud_train.shape[1]),
        "support_signature_dim": int(sig_train.shape[1]),
        "template_prob_dim": int(prob_train.shape[1]),
    }
    return feat_train, feat_test, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--scene-cache", default=ASSETS + "/official_pointcloud_scene_cache.npz")
    ap.add_argument("--base", default=ASSETS + "/cache_v125/v125_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=ASSETS + "/cache_v128/v128_Round1_Test_Channel.npy")
    ap.add_argument("--truth-support", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--train-base-ad", default=ASSETS + "/train_v125eq_crossfit_base_ad.npy")
    ap.add_argument("--train-base-support-out", default=ASSETS + "/cache_v128/train_v125eq_collapsed_support.npy")
    ap.add_argument("--current-target", default=ASSETS + "/cache_v124/v124_visibility_misleading_overdrive_target.npy")
    ap.add_argument("--v121-target", default=ASSETS + "/cache_v121/v121_support_template_manifold_target.npy")
    ap.add_argument("--v123-target", default=ASSETS + "/cache_v123/v123_visibility_aware_template_target.npy")
    ap.add_argument("--v124-target", default=ASSETS + "/cache_v124/v124_visibility_misleading_overdrive_target.npy")
    ap.add_argument("--train-template-probs", default=ASSETS + "/train_v125eq_crossfit_template_probs.npy")
    ap.add_argument("--test-template-probs", default=ASSETS + "/cache_v124/v124_visibility_misleading_overdrive_probs.npy")
    ap.add_argument("--train-total-power", default=ASSETS + "/train_total_power.npy")
    ap.add_argument("--min-train-power", type=float, default=-1.0)
    ap.add_argument("--target-out", default=ASSETS + "/cache_v128/v128_badpoint_support_residual_target.npy")
    ap.add_argument("--weights-out", default=ASSETS + "/cache_v128/v128_badpoint_support_residual_weights.npy")
    ap.add_argument("--prob-out", default=ASSETS + "/cache_v128/v128_badpoint_support_residual_probs.npy")
    ap.add_argument("--label-out", default=ASSETS + "/cache_v128/v128_badpoint_residual_train_labels.npy")
    ap.add_argument("--report", default=ASSETS + "/v128_badpoint_support_residual_templates_generate.json")
    ap.add_argument("--zip-out", default=str(PATHS.round1_versions / "v128_badpoint_support_residual_templates_candidate_pending.zip"))
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ad-batch", type=int, default=10)
    ap.add_argument("--project-batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12807)
    ap.add_argument("--bad-quantile", type=float, default=0.58)
    ap.add_argument("--min-bad-cos", type=float, default=0.965)
    ap.add_argument("--clusters", type=int, default=72)
    ap.add_argument("--svd-components", type=int, default=72)
    ap.add_argument("--ratio-clip", default="0.20,5.20")
    ap.add_argument("--k-bad", type=int, default=34)
    ap.add_argument("--bad-power", type=float, default=3.15)
    ap.add_argument("--bad-floor", type=float, default=0.42)
    ap.add_argument("--proto-temp", type=float, default=0.018)
    ap.add_argument("--bad-weight", type=float, default=0.76)
    ap.add_argument("--proto-weight", type=float, default=0.24)
    ap.add_argument("--eta", default="clear:0.0002,light:0.004,medium:0.560,heavy:0.780")
    ap.add_argument("--min-gate", type=float, default=0.25)
    ap.add_argument("--max-gate", type=float, default=3.00)
    ap.add_argument("--conf-center", type=float, default=0.20)
    ap.add_argument("--conf-width", type=float, default=0.20)
    ap.add_argument("--entropy-low", type=float, default=0.20)
    ap.add_argument("--entropy-high", type=float, default=0.98)
    ap.add_argument("--gain-clip", default="0.36,3.80")
    ap.add_argument("--top-gain-clip", default="0.46,2.90")
    ap.add_argument("--top-components", type=int, default=2816)
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    args = ap.parse_args()
    args.target_floor = float(args.floor)

    data = Path(args.data)
    pos_train = np.load(data / "Round1_Train_Pos.npy")
    pos_test = np.load(data / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    blocked_test = geo["test_blocked"].astype(np.float64)
    regimes_train = regime_of(blocked_train)
    regimes_test = regime_of(blocked_test)
    scene = np.load(args.scene_cache, allow_pickle=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    truth = normalize_support(np.load(args.truth_support, mmap_mode="r"), float(args.floor))
    train_base = collapsed_support_from_ad(args.train_base_ad, args.train_base_support_out, float(args.floor), int(args.ad_batch))
    current = normalize_support(np.load(args.current_target), float(args.floor))
    v121 = normalize_support(np.load(args.v121_target), float(args.floor))
    v123 = normalize_support(np.load(args.v123_target), float(args.floor))
    v124 = normalize_support(np.load(args.v124_target), float(args.floor))
    train_template_prob = np.load(args.train_template_probs).astype(np.float32)
    test_template_prob = np.load(args.test_template_probs).astype(np.float32)

    bad_mask, train_cos = build_bad_mask(
        train_base,
        truth,
        regimes_train,
        float(args.bad_quantile),
        float(args.min_bad_cos),
    )
    excluded_power_count = 0
    excluded_power_by_regime = {r: 0 for r in REGIMES}
    if float(args.min_train_power) >= 0 and Path(args.train_total_power).exists():
        train_power = np.load(args.train_total_power)
        exclude_power = train_power <= float(args.min_train_power)
        excluded_power_count = int(np.sum(bad_mask & exclude_power))
        excluded_power_by_regime = {r: int(np.sum((bad_mask & exclude_power) & (regimes_train == r))) for r in REGIMES}
        bad_mask = bad_mask & (~exclude_power)
        if int(bad_mask.sum()) < 256:
            raise ValueError("too few bad points remain after min-train-power filtering")
    ratio_clip = tuple(float(x) for x in args.ratio_clip.split(","))
    proto_ratio, proto_truth, labels, template_report = residual_ratio_templates(
        train_base,
        truth,
        bad_mask,
        int(args.clusters),
        int(args.svd_components),
        int(args.seed),
        float(args.floor),
        ratio_clip,
    )

    feat_train, feat_test, feature_report = build_features(
        pos_train,
        pos_test,
        blocked_train,
        blocked_test,
        train_base,
        current,
        train_template_prob,
        test_template_prob,
        scene,
        float(args.floor),
    )
    bad_feat = feat_train[bad_mask]
    idx_bad, dist_bad = pairwise_knn(feat_test, bad_feat, int(args.k_bad), exclude_self=False)
    w_bad = weights_from_dist(dist_bad, float(args.bad_power), float(args.bad_floor))
    prob_bad = probability_from_bad_neighbors(idx_bad, w_bad, labels, proto_ratio.shape[0])

    proto_flat = proto_truth.reshape(proto_truth.shape[0], -1).astype(np.float32)
    cur_flat = current.reshape(len(current), -1).astype(np.float32)
    proto_norm = proto_flat / np.maximum(np.linalg.norm(proto_flat, axis=1, keepdims=True), TINY)
    cur_norm = cur_flat / np.maximum(np.linalg.norm(cur_flat, axis=1, keepdims=True), TINY)
    prob_proto = softmax_np(cur_norm @ proto_norm.T, temp=float(args.proto_temp))

    bw = float(args.bad_weight)
    pw = float(args.proto_weight)
    prob = (bw * prob_bad + pw * prob_proto) / max(bw + pw, 1e-6)
    prob /= np.maximum(prob.sum(axis=1, keepdims=True), TINY)
    ratio = mixture_ratio(proto_ratio, prob, float(args.floor), int(args.batch))
    residual_target = apply_ratio_support(current, ratio, float(args.floor))

    maxw = prob.max(axis=1).astype(np.float32)
    ent = entropy01(prob).astype(np.float32)
    avg_badness = (w_bad * (1.0 - train_cos[bad_mask][idx_bad])).sum(axis=1).astype(np.float32)
    residual_cos_current = row_cos(residual_target, current).astype(np.float32)
    residual_strength = np.clip((avg_badness - 0.22) / 0.36, 0.0, 1.0)
    residual_strength *= (0.42 + 0.58 * np.clip((maxw - 0.12) / 0.28, 0.0, 1.0)).astype(np.float32)
    residual_strength *= (0.50 + 0.50 * np.clip((0.94 - ent) / 0.45, 0.0, 1.0)).astype(np.float32)
    # Do not require a large residual/current disagreement.  The previous
    # conservative gate made the final target almost identical to v125, which
    # cannot test whether bad-point residual templates contain new signal.
    residual_strength *= (0.78 + 0.22 * np.clip((1.0 - residual_cos_current) / 0.035, 0.0, 1.0)).astype(np.float32)

    regime_scale = np.array(
        [
            {"clear": 0.02, "light": 0.18, "medium": 0.86, "heavy": 1.08}[str(r)]
            for r in regimes_test
        ],
        dtype=np.float32,
    )
    residual_strength = np.clip(residual_strength * regime_scale, 0.0, 1.0).astype(np.float32)

    # Blend the residual patch target with already verified support anchors.
    # v121/v123/v124 keep the new patch from taking over when it disagrees with
    # the existing proven manifold.
    coeff = np.zeros((len(current), 5), dtype=np.float32)
    coeff[:, 0] = 1.0 - residual_strength
    coeff[:, 1] = 0.82 * residual_strength
    coeff[:, 2] = 0.10 * residual_strength
    coeff[:, 3] = 0.055 * residual_strength
    coeff[:, 4] = 0.025 * residual_strength
    target = log_blend([current, residual_target, v124, v123, v121], coeff, float(args.floor), int(args.batch))

    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.target_out, target.astype(np.float32))
    weights = topk_prob(prob, 12).astype(np.float32)
    np.save(args.weights_out, weights)
    np.save(args.prob_out, prob.astype(np.float32))
    np.save(args.label_out, labels.astype(np.int64))

    proj = projection(target, weights, regimes_test, pos_test, args)
    payload = {
        "method": "v128_badpoint_support_residual_templates",
        "base": args.base,
        "operation": "mine train bad-point residual support templates from truth-v125eq crossfit; select residual patches by official point-cloud geometry/current support/template probability; support-only projection preserving v125 phase/total power/pol-UE allocation",
        "args": vars(args),
        "bad_train": {
            "train_base_vs_truth_cos": summarize(train_cos),
            "bad_mask_count": int(bad_mask.sum()),
            "bad_mask_fraction": float(np.mean(bad_mask)),
            "excluded_power_count": int(excluded_power_count),
            "excluded_power_by_regime": excluded_power_by_regime,
            "bad_cos": summarize(train_cos[bad_mask]),
            "bad_regime_counts": {r: int(np.sum(regimes_train[bad_mask] == r)) for r in REGIMES},
        },
        "templates": template_report,
        "features": feature_report,
        "selector": {
            "prob_top1": summarize(maxw),
            "prob_entropy": summarize(ent),
            "avg_badness": summarize(avg_badness),
            "residual_target_vs_current": summarize(residual_cos_current),
            "residual_strength": summarize(residual_strength),
            "nearest_bad_dist": summarize(np.sqrt(np.maximum(dist_bad[:, 0], 0.0))),
        },
        "target_cos": {
            "target_vs_current": summarize(row_cos(target, current)),
            "target_vs_residual": summarize(row_cos(target, residual_target)),
            "target_vs_v124": summarize(row_cos(target, v124)),
            "target_vs_v123": summarize(row_cos(target, v123)),
            "target_vs_v121": summarize(row_cos(target, v121)),
        },
        "projection": proj,
        "outputs": {
            "submission_npy": args.output,
            "target": args.target_out,
            "weights": args.weights_out,
            "prob": args.prob_out,
            "labels": args.label_out,
            "train_base_support": args.train_base_support_out,
        },
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    files = [
        (args.output, "Round1_Test_Channel.npy"),
        (__file__, "generate_v128_badpoint_support_residual_templates.py"),
        (ASSETS + "/generate_v121_support_template_manifold.py", "generate_v121_support_template_manifold.py"),
        (ASSETS + "/generate_v115_pointcloud_scene_support.py", "generate_v115_pointcloud_scene_support.py"),
        (ASSETS + "/generate_v112_kernel_moe_path_support.py", "generate_v112_kernel_moe_path_support.py"),
        (ASSETS + "/generate_v98_path_ensemble_graph_attention_joint_ad.py", "generate_v98_path_ensemble_graph_attention_joint_ad.py"),
        (ASSETS + "/generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
        (ASSETS + "/powerfield_common.py", "powerfield_common.py"),
        (PATHS.code_root / "project_paths.py", "project_paths.py"),
        (ASSETS + "/build_official_pointcloud_scene.py", "build_official_pointcloud_scene.py"),
        (ASSETS + "/build_v125_equivalent_crossfit_base.py", "build_v125_equivalent_crossfit_base.py"),
    ]
    Path(args.zip_out).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_STORED) as zf:
        for src, arc in files:
            zf.write(src, arc)
    print(f"wrote {args.zip_out}", flush=True)


if __name__ == "__main__":
    main()
