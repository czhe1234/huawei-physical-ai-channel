#!/usr/bin/env python3
"""Generate v124: visibility misleading overdrive.

v123 confirmed that occlusion-aware visibility helps the support-template
selector.  This version is a more aggressive follow-up, but it is still
targeted: it mainly acts on points where raw mirror-reflection candidates look
strong while the occlusion test says those paths are blocked.

* base is v123;
* keep phase, complex scalar structure, and per-point total power from v123;
* rebuild the support-template selector with occlusion-aware visibility
  features derived from the official point-cloud grid;
* do not directly use mirror-reflection candidates as paths unless their
  BS->reflection and reflection->UE segments pass a lightweight blockage test.

The purpose is to fix the diagnostic issue where simple reflection rays crossed
buildings.  Geometry is used as a selector for propagation-mode templates, not
as a direct ray-tracing reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import BS_POS, MH, MV, S_SC, summarize
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
    build_prototypes,
    cluster_embeddings,
    fit_support_svd,
    support_signature,
    template_blend,
    topk_prob,
)
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


REGIMES = ("clear", "light", "medium", "heavy")


def segment_blockage_many(
    a: np.ndarray,
    b: np.ndarray,
    scene: np.lib.npyio.NpzFile,
    samples: int,
    vertical_threshold: int,
    z_margin: float,
    t0: float,
    t1: float,
    batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Occlusion test for many straight segments against the point-cloud grid."""

    n = len(a)
    blocked_out = np.zeros(n, dtype=bool)
    hit_out = np.zeros(n, dtype=np.int16)
    clearance_out = np.full(n, np.inf, dtype=np.float32)

    x_min = float(scene["x_min"])
    y_min = float(scene["y_min"])
    res = float(scene["resolution"])
    nx = int(scene["grid_nx"])
    ny = int(scene["grid_ny"])
    z_max_grid = scene["grid_z_max"]
    vertical_grid = scene["grid_vertical_count"]
    ts = np.linspace(float(t0), float(t1), int(samples), dtype=np.float64)

    for start in range(0, n, int(batch)):
        end = min(start + int(batch), n)
        aa = a[start:end].astype(np.float64)
        bb = b[start:end].astype(np.float64)
        pts = aa[:, None, :] + ts[None, :, None] * (bb[:, None, :] - aa[:, None, :])
        ix = np.floor((pts[..., 0] - x_min) / res).astype(np.int64)
        iy = np.floor((pts[..., 1] - y_min) / res).astype(np.int64)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        z_ray = pts[..., 2]

        z_cell = np.full(ix.shape, -np.inf, dtype=np.float64)
        vertical = np.zeros(ix.shape, dtype=np.int32)
        if np.any(ok):
            z_cell[ok] = z_max_grid[iy[ok], ix[ok]].astype(np.float64)
            vertical[ok] = vertical_grid[iy[ok], ix[ok]].astype(np.int32)
        blocked = ok & (vertical >= int(vertical_threshold)) & np.isfinite(z_cell) & (z_cell >= z_ray - float(z_margin))
        blocked_out[start:end] = blocked.any(axis=1)
        hit_out[start:end] = blocked.sum(axis=1).astype(np.int16)
        clearance = z_ray - z_cell
        clearance[~ok] = np.inf
        clearance_out[start:end] = np.nanmin(clearance, axis=1).astype(np.float32)

    return blocked_out, hit_out, clearance_out


def visibility_features(
    pos: np.ndarray,
    blocked: np.ndarray,
    scene: np.lib.npyio.NpzFile,
    prefix: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build visibility features using LOS stats plus occluded reflection checks."""

    features = scene[f"{prefix}_features"].astype(np.float32)
    names = [str(x) for x in scene["feature_names"]]
    name_to_i = {name: i for i, name in enumerate(names)}

    def col(name: str, default: float = 0.0) -> np.ndarray:
        if name not in name_to_i:
            return np.full(len(pos), default, dtype=np.float32)
        return features[:, name_to_i[name]].astype(np.float32)

    los_frac = col("pc_los_height_block_frac")
    los_count = col("pc_los_height_block_count")
    los_surplus_sum = col("pc_los_height_surplus_sum")
    los_surplus_max = col("pc_los_height_surplus_max")
    los_first_t = col("pc_los_height_first_t", 1.0)
    los_last_t = col("pc_los_height_last_t", 0.0)
    local_min = col("pc_local_ray_min", 80.0)
    local_p10 = col("pc_local_ray_p10", 80.0)
    local_med = col("pc_local_ray_median", 80.0)
    local_std = col("pc_local_ray_std")
    nn_min = col("pc_los_nn_min_dist", 8.0)
    nn_p10 = col("pc_los_nn_p10_dist", 8.0)
    nn_inv = col("pc_los_nn_mean_inv")

    score = scene[f"{prefix}_reflection_score"].astype(np.float32)
    path_len = scene[f"{prefix}_reflection_path_length"].astype(np.float32)
    excess = scene[f"{prefix}_reflection_excess_length"].astype(np.float32)
    refl = scene[f"{prefix}_reflection_point"].astype(np.float32)
    patch = scene[f"{prefix}_reflection_patch_id"].astype(np.int64)
    n, k = score.shape

    valid = (patch >= 0) & np.isfinite(score) & (score > float(args.min_reflection_score))
    flat_valid = valid.reshape(-1)
    rx_rep = np.repeat(pos.astype(np.float64), k, axis=0)
    rp = refl.reshape(-1, 3).astype(np.float64)
    bs_rep = np.repeat(BS_POS[None, :].astype(np.float64), n * k, axis=0)

    b1, h1, c1 = segment_blockage_many(
        bs_rep,
        rp,
        scene,
        int(args.occlusion_samples),
        int(args.vertical_threshold),
        float(args.z_margin),
        0.02,
        0.90,
        int(args.occlusion_batch),
    )
    b2, h2, c2 = segment_blockage_many(
        rp,
        rx_rep,
        scene,
        int(args.occlusion_samples),
        int(args.vertical_threshold),
        float(args.z_margin),
        0.10,
        0.98,
        int(args.occlusion_batch),
    )
    ref_blocked = (b1 | b2).reshape(n, k)
    ref_clear = valid & (~ref_blocked)
    ref_block_hits = (h1 + h2).reshape(n, k).astype(np.float32)
    ref_clearance = np.minimum(c1, c2).reshape(n, k).astype(np.float32)

    raw_score = np.where(valid, score, 0.0)
    clear_score = np.where(ref_clear, score, 0.0)
    blocked_score = np.where(valid & ref_blocked, score, 0.0)
    top_raw_score = raw_score.max(axis=1)
    top_clear_score = clear_score.max(axis=1)
    top_blocked_score = blocked_score.max(axis=1)
    clear_count = ref_clear.sum(axis=1).astype(np.float32)
    valid_count = valid.sum(axis=1).astype(np.float32)
    blocked_count = (valid & ref_blocked).sum(axis=1).astype(np.float32)
    clear_ratio = clear_count / np.maximum(valid_count, 1.0)
    clear_score_sum = clear_score.sum(axis=1)
    blocked_score_sum = blocked_score.sum(axis=1)
    misleading_reflection = np.clip(top_blocked_score - top_clear_score, 0.0, 1.0)

    best_clear_id = np.argmax(clear_score, axis=1)
    row = np.arange(n)
    has_clear = top_clear_score > 0.0
    best_clear_len = np.where(has_clear, path_len[row, best_clear_id], 0.0).astype(np.float32)
    best_clear_excess = np.where(has_clear, excess[row, best_clear_id], 0.0).astype(np.float32)
    best_clear_hits = np.where(has_clear, ref_block_hits[row, best_clear_id], 0.0).astype(np.float32)
    best_clear_margin = np.where(has_clear, ref_clearance[row, best_clear_id], -5.0).astype(np.float32)

    los_clear = ((blocked <= 0) & (los_frac < 0.02)).astype(np.float32)
    no_visible_path = ((los_clear < 0.5) & (clear_count <= 0)).astype(np.float32)

    out = np.column_stack(
        [
            los_clear,
            np.clip(los_frac, 0.0, 1.0),
            np.log1p(np.maximum(los_count, 0.0)) / 5.0,
            np.log1p(np.maximum(los_surplus_sum, 0.0)) / 5.0,
            np.log1p(np.maximum(los_surplus_max, 0.0)) / 4.0,
            np.clip(los_first_t, 0.0, 1.0),
            np.clip(los_last_t, 0.0, 1.0),
            np.clip(local_min, 0.0, 80.0) / 80.0,
            np.clip(local_p10, 0.0, 80.0) / 80.0,
            np.clip(local_med, 0.0, 80.0) / 80.0,
            np.clip(local_std, 0.0, 50.0) / 50.0,
            np.clip(nn_min, 0.0, 10.0) / 10.0,
            np.clip(nn_p10, 0.0, 10.0) / 10.0,
            np.log1p(np.maximum(nn_inv, 0.0)) / 5.0,
            valid_count / max(k, 1),
            clear_count / max(k, 1),
            blocked_count / max(k, 1),
            clear_ratio,
            top_raw_score,
            top_clear_score,
            top_blocked_score,
            clear_score_sum,
            blocked_score_sum,
            misleading_reflection,
            no_visible_path,
            np.log1p(np.maximum(best_clear_len, 0.0)) / 7.0,
            np.log1p(np.maximum(best_clear_excess, 0.0)) / 6.0,
            np.clip(best_clear_hits, 0.0, 40.0) / 40.0,
            np.clip(best_clear_margin, -5.0, 12.0) / 12.0,
        ]
    ).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)

    report = {
        f"{prefix}_los_clear": summarize(los_clear),
        f"{prefix}_valid_ref_count": summarize(valid_count),
        f"{prefix}_clear_ref_count": summarize(clear_count),
        f"{prefix}_top_raw_ref_score": summarize(top_raw_score),
        f"{prefix}_top_clear_ref_score": summarize(top_clear_score),
        f"{prefix}_top_blocked_ref_score": summarize(top_blocked_score),
        f"{prefix}_misleading_reflection": summarize(misleading_reflection),
        f"{prefix}_no_visible_path": summarize(no_visible_path),
    }
    aux = {
        "top_raw_score": top_raw_score.astype(np.float32),
        "top_clear_score": top_clear_score.astype(np.float32),
        "top_blocked_score": top_blocked_score.astype(np.float32),
        "misleading_reflection": misleading_reflection.astype(np.float32),
        "no_visible_path": no_visible_path.astype(np.float32),
        "los_clear": los_clear.astype(np.float32),
        "clear_ref_count": clear_count.astype(np.float32),
    }
    return out, {"summary": report, "aux": aux}


def probability_from_labels(idx: np.ndarray, weights: np.ndarray, labels: np.ndarray, clusters: int) -> np.ndarray:
    eye = np.eye(clusters, dtype=np.float32)
    prob = (weights[:, :, None] * eye[labels[idx]]).sum(axis=1).astype(np.float32)
    prob /= np.maximum(prob.sum(axis=1, keepdims=True), TINY)
    return prob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--scene-cache", default=ASSETS + "/official_pointcloud_scene_cache.npz")
    ap.add_argument("--base", default=ASSETS + "/cache_v123/v123_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=ASSETS + "/cache_v124/v124_Round1_Test_Channel.npy")
    ap.add_argument("--truth-support", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--train-current-support", default=ASSETS + "/train_v98like_loo_joint_target.npy")
    ap.add_argument("--current-target", default=ASSETS + "/cache_v123/v123_visibility_aware_template_target.npy")
    ap.add_argument("--v115-anchor", default=ASSETS + "/cache_v115/v115_pointcloud_scene_support_target.npy")
    ap.add_argument("--v112-anchor", default=ASSETS + "/cache_v112/v112_kernel_moe_path_support_target.npy")
    ap.add_argument("--target-out", default=ASSETS + "/cache_v124/v124_visibility_misleading_overdrive_target.npy")
    ap.add_argument("--weights-out", default=ASSETS + "/cache_v124/v124_visibility_misleading_overdrive_weights.npy")
    ap.add_argument("--prob-out", default=ASSETS + "/cache_v124/v124_visibility_misleading_overdrive_probs.npy")
    ap.add_argument("--report", default=ASSETS + "/v124_visibility_misleading_overdrive_generate.json")
    ap.add_argument("--zip-out", default=str(PATHS.round1_versions / "v124_visibility_misleading_overdrive_candidate_pending.zip"))
    ap.add_argument("--clusters", type=int, default=96)
    ap.add_argument("--svd-components", type=int, default=72)
    ap.add_argument("--seed", type=int, default=12413)
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
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project-batch", type=int, default=10)
    ap.add_argument("--eta", default="clear:0.00015,light:0.0025,medium:0.720,heavy:1.020")
    ap.add_argument("--min-gate", type=float, default=0.20)
    ap.add_argument("--max-gate", type=float, default=2.55)
    ap.add_argument("--conf-center", type=float, default=0.17)
    ap.add_argument("--conf-width", type=float, default=0.15)
    ap.add_argument("--entropy-low", type=float, default=0.20)
    ap.add_argument("--entropy-high", type=float, default=0.92)
    ap.add_argument("--gain-clip", default="0.36,3.75")
    ap.add_argument("--top-gain-clip", default="0.50,2.75")
    ap.add_argument("--top-components", type=int, default=2816)
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    ap.add_argument("--occlusion-samples", type=int, default=120)
    ap.add_argument("--occlusion-batch", type=int, default=4096)
    ap.add_argument("--vertical-threshold", type=int, default=2)
    ap.add_argument("--z-margin", type=float, default=0.35)
    ap.add_argument("--min-reflection-score", type=float, default=1e-5)
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

    print("load supports", flush=True)
    truth = normalize_support(np.load(args.truth_support, mmap_mode="r"), float(args.floor))
    train_current = normalize_support(np.load(args.train_current_support, mmap_mode="r"), float(args.floor))
    current = normalize_support(np.load(args.current_target), float(args.floor))
    v115_anchor = normalize_support(np.load(args.v115_anchor), float(args.floor))
    v112_anchor = normalize_support(np.load(args.v112_anchor), float(args.floor))

    label_path = ASSETS / "cache_v121/v121_support_template_manifold_train_labels.npy"
    if label_path.exists():
        labels = np.load(label_path).astype(np.int64)
        k = int(labels.max()) + 1
        prototypes = build_prototypes(truth, labels, k, float(args.floor))
    else:
        support_x = np.log(np.maximum(truth.reshape(len(truth), -1), float(args.floor))).astype(np.float32)
        emb, _, _ = fit_support_svd(support_x, int(args.svd_components), int(args.seed))
        labels = cluster_embeddings(emb, int(args.clusters), int(args.seed))
        k = int(args.clusters)
        prototypes = build_prototypes(truth, labels, k, float(args.floor))

    print("build official scene/support/visibility features", flush=True)
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
    vis_train_raw, vis_train_info = visibility_features(pos_train, blocked_train, scene, "train", args)
    vis_test_raw, vis_test_info = visibility_features(pos_test, blocked_test, scene, "test", args)
    sig_train_raw = support_signature(train_current, float(args.floor))
    sig_test_raw = support_signature(current, float(args.floor))
    spatial_train_raw = spatial_embedding(pos_train, blocked_train)
    spatial_test_raw = spatial_embedding(pos_test, blocked_test)

    cloud_train, cloud_test = standardize(cloud_train_raw, cloud_test_raw)
    vis_train, vis_test = standardize(vis_train_raw, vis_test_raw)
    sig_train, sig_test = standardize(sig_train_raw, sig_test_raw)
    spatial_train, spatial_test = standardize(spatial_train_raw, spatial_test_raw)
    scene_train = np.concatenate([cloud_train, 0.52 * sig_train, 0.95 * vis_train], axis=1).astype(np.float32)
    scene_test = np.concatenate([cloud_test, 0.52 * sig_test, 0.95 * vis_test], axis=1).astype(np.float32)
    geom_train = np.concatenate([spatial_train, 0.78 * sig_train, 0.35 * vis_train], axis=1).astype(np.float32)
    geom_test = np.concatenate([spatial_test, 0.78 * sig_test, 0.35 * vis_test], axis=1).astype(np.float32)
    vis_bank_train = np.concatenate([1.55 * vis_train, 0.38 * sig_train, 0.18 * spatial_train], axis=1).astype(np.float32)
    vis_bank_test = np.concatenate([1.55 * vis_test, 0.38 * sig_test, 0.18 * spatial_test], axis=1).astype(np.float32)

    print("predict visibility-aware template distribution", flush=True)
    idx_scene, dist_scene = pairwise_knn(scene_test, scene_train, int(args.k_scene), exclude_self=False)
    idx_geom, dist_geom = pairwise_knn(geom_test, geom_train, int(args.k_geom), exclude_self=False)
    idx_vis, dist_vis = pairwise_knn(vis_bank_test, vis_bank_train, int(args.k_vis), exclude_self=False)
    w_scene = weights_from_dist(dist_scene, float(args.scene_power), float(args.scene_floor))
    w_geom = weights_from_dist(dist_geom, float(args.geom_power), float(args.geom_floor))
    w_vis = weights_from_dist(dist_vis, float(args.vis_power), float(args.vis_floor))
    prob_scene = probability_from_labels(idx_scene, w_scene, labels, k)
    prob_geom = probability_from_labels(idx_geom, w_geom, labels, k)
    prob_vis = probability_from_labels(idx_vis, w_vis, labels, k)

    proto_flat = prototypes.reshape(k, -1).astype(np.float32)
    cur_flat = current.reshape(len(current), -1).astype(np.float32)
    proto_norm = proto_flat / np.maximum(np.linalg.norm(proto_flat, axis=1, keepdims=True), TINY)
    cur_norm = cur_flat / np.maximum(np.linalg.norm(cur_flat, axis=1, keepdims=True), TINY)
    prob_cur = softmax_np(cur_norm @ proto_norm.T, temp=float(args.proto_temp))

    sw = float(args.scene_weight)
    gw = float(args.geom_weight)
    vw = float(args.vis_weight)
    cw = float(args.current_weight)
    prob = (sw * prob_scene + gw * prob_geom + vw * prob_vis + cw * prob_cur) / max(sw + gw + vw + cw, 1e-6)
    prob /= np.maximum(prob.sum(axis=1, keepdims=True), TINY)
    template_target = template_blend(prototypes, prob, float(args.floor), int(args.batch))

    scene_vis_agree = np.sum(np.sqrt(np.maximum(prob_scene, 0.0) * np.maximum(prob_vis, 0.0)), axis=1).astype(np.float32)
    vis_cur_agree = np.sum(np.sqrt(np.maximum(prob_vis, 0.0) * np.maximum(prob_cur, 0.0)), axis=1).astype(np.float32)
    template_cos_current = row_cos(template_target, current).astype(np.float32)
    prob_top = topk_prob(prob, 8)[:, 0].astype(np.float32)
    prob_ent = entropy01(prob).astype(np.float32)
    aux = vis_test_info["aux"]
    visible_signal = np.clip(aux["top_clear_score"] / 0.55, 0.0, 1.0).astype(np.float32)
    misleading_signal = np.clip(aux["misleading_reflection"] / 0.55, 0.0, 1.0).astype(np.float32)
    blocked_no_path = aux["no_visible_path"].astype(np.float32)

    divergence = np.clip((1.0 - template_cos_current - 0.030) / 0.115, 0.0, 1.0).astype(np.float32)
    misleading_overdrive = (
        np.clip((misleading_signal - 0.42) / 0.48, 0.0, 1.0)
        * (0.42 + 0.58 * divergence)
        * (0.62 + 0.38 * np.clip((prob_top - 0.18) / 0.35, 0.0, 1.0))
    ).astype(np.float32)

    reg_strength = {"clear": 0.004, "light": 0.020, "medium": 0.660, "heavy": 0.930}
    strength = np.array([reg_strength[str(r)] for r in regimes_test], dtype=np.float32)
    strength *= (0.48 + 0.52 * np.clip((prob_top - 0.12) / 0.28, 0.0, 1.0)).astype(np.float32)
    strength *= (0.54 + 0.46 * np.clip((scene_vis_agree - 0.18) / 0.36, 0.0, 1.0)).astype(np.float32)
    strength *= (0.58 + 0.42 * np.clip((vis_cur_agree - 0.22) / 0.36, 0.0, 1.0)).astype(np.float32)
    strength *= (0.58 + 0.42 * np.clip((template_cos_current - 0.84) / 0.12, 0.0, 1.0)).astype(np.float32)
    # If a high-score mirror candidate is actually blocked, v121's raw
    # reflection-like geometry can be misleading; allow a stronger template
    # correction in those medium/heavy cases.  If there is simply no visible
    # path and no misleading candidate, be more conservative.
    strength *= (0.80 + 0.22 * misleading_signal + 0.10 * visible_signal + 0.36 * misleading_overdrive).astype(np.float32)
    strength *= (1.0 - 0.16 * blocked_no_path * (1.0 - misleading_signal)).astype(np.float32)
    strength = np.clip(strength, 0.0, 1.02).astype(np.float32)

    coeff = np.zeros((len(current), 4), dtype=np.float32)
    coeff[:, 0] = 1.0 - strength
    coeff[:, 1] = 0.91 * strength
    coeff[:, 2] = 0.065 * strength
    coeff[:, 3] = 0.025 * strength
    target = log_blend([current, template_target, v115_anchor, v112_anchor], coeff, float(args.floor), int(args.batch))
    weights = topk_prob(prob, 12)

    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.target_out, target)
    np.save(args.weights_out, weights.astype(np.float32))
    np.save(args.prob_out, prob.astype(np.float32))

    proj = projection(target, weights.astype(np.float32), regimes_test, pos_test, args)
    payload = {
        "method": "v124_visibility_misleading_overdrive",
        "base": args.base,
        "operation": "v123 base; targeted overdrive for misleading blocked-reflection points in occlusion-aware support-template selector; preserve phase/scalar/total power",
        "args": vars(args),
        "visibility": {
            "train": vis_train_info["summary"],
            "test": vis_test_info["summary"],
            "test_visible_signal": summarize(visible_signal),
            "test_misleading_signal": summarize(misleading_signal),
            "test_blocked_no_path": summarize(blocked_no_path),
            "test_misleading_overdrive": summarize(misleading_overdrive),
        },
        "template": {
            "scene_vis_agreement": summarize(scene_vis_agree),
            "vis_cur_agreement": summarize(vis_cur_agree),
            "prob_top1": summarize(prob_top),
            "prob_entropy": summarize(prob_ent),
            "template_vs_current": summarize(template_cos_current),
            "projection_strength": summarize(strength),
        },
        "target_cos": {
            "target_vs_current": summarize(row_cos(target, current)),
            "template_vs_current": summarize(row_cos(template_target, current)),
            "target_vs_template": summarize(row_cos(target, template_target)),
            "target_vs_v115": summarize(row_cos(target, v115_anchor)),
            "target_vs_v112": summarize(row_cos(target, v112_anchor)),
        },
        "projection": proj,
        "outputs": {
            "submission_npy": args.output,
            "target": args.target_out,
            "weights": args.weights_out,
            "prob": args.prob_out,
        },
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    files = [
        (args.output, "Round1_Test_Channel.npy"),
        (__file__, "generate_v124_visibility_misleading_overdrive.py"),
        (ASSETS + "/generate_v121_support_template_manifold.py", "generate_v121_support_template_manifold.py"),
        (ASSETS + "/generate_v115_pointcloud_scene_support.py", "generate_v115_pointcloud_scene_support.py"),
        (ASSETS + "/generate_v112_kernel_moe_path_support.py", "generate_v112_kernel_moe_path_support.py"),
        (ASSETS + "/generate_v98_path_ensemble_graph_attention_joint_ad.py", "generate_v98_path_ensemble_graph_attention_joint_ad.py"),
        (ASSETS + "/generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
        (ASSETS + "/powerfield_common.py", "powerfield_common.py"),
        (PATHS.code_root / "project_paths.py", "project_paths.py"),
        (ASSETS + "/build_official_pointcloud_scene.py", "build_official_pointcloud_scene.py"),
    ]
    Path(args.zip_out).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_STORED) as zf:
        for src, arc in files:
            zf.write(src, arc)
    print(f"wrote {args.zip_out}", flush=True)


if __name__ == "__main__":
    main()
