#!/usr/bin/env python3
"""Generate v115: official point-cloud scene support target.

This is the stronger official point-cloud replacement test for the Sionna
path-token branch.

It does not read ``sionna_path_candidates.npz`` and does not use mesh/object/
material/triangle labels.  The new support target is built only from official
data-derived information:

* UE coordinates and fixed BS position;
* continuous point-cloud scene cache from ``Round1_Map.ply``;
* wall patches, corridor features, and official point-cloud specular candidates;
* height/blocked regime from the official point cloud;
* train-channel collapsed AD support;
* test-test relations computed from the same official geometry features.

The complex channel backbone is v112 because v112 is the current best online
submission.  The experiment measures whether an official-cloud geometry target
can replace or complement the Sionna path-token target at the support stage.
As usual in the v102+ branch, phase, complex scalar structure, and per-point
total power are preserved; only collapsed joint AD support is nudged.
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
    fit_moe_gate,
    kernel_conf,
    log_average,
    log_blend,
    normalize_support,
    pairwise_knn,
    row_cos,
    softmax_np,
    standardize,
    weights_from_dist,
)
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


REGIMES = ("clear", "light", "medium", "heavy")


def official_scene_embedding(
    pos: np.ndarray,
    blocked: np.ndarray,
    scene_feat: np.ndarray,
    refl_score: np.ndarray,
    refl_path: np.ndarray,
    refl_excess: np.ndarray,
    refl_point: np.ndarray,
    refl_patch_id: np.ndarray,
    wall_patches: np.ndarray,
) -> np.ndarray:
    """Build official-data-only scene features.

    ``scene_feat`` is produced by ``build_official_pointcloud_scene.py`` from
    the official point cloud.  Reflection candidates and wall patches are also
    derived from the same point cloud.  No Sionna mesh labels, object IDs,
    material IDs, or path-token files are used.
    """
    pos = np.asarray(pos, dtype=np.float32)
    f = np.nan_to_num(np.asarray(scene_feat, dtype=np.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    rel = pos - BS_POS[None, :].astype(np.float32)
    dist = np.linalg.norm(rel, axis=1)
    horiz = np.maximum(np.hypot(rel[:, 0], rel[:, 1]), 1e-6)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    el = np.arctan2(rel[:, 2], horiz)

    reg = regime_of(blocked)
    reg_oh = np.stack([(reg == r).astype(np.float32) for r in REGIMES], axis=1)

    f_raw = f
    f_log = np.log1p(np.maximum(f, 0.0)).astype(np.float32)

    score = np.nan_to_num(refl_score.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    path = np.nan_to_num(refl_path.astype(np.float32), nan=0.0, posinf=900.0, neginf=0.0)
    excess = np.nan_to_num(refl_excess.astype(np.float32), nan=0.0, posinf=900.0, neginf=0.0)
    point = np.nan_to_num(refl_point.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    patch_id = np.asarray(refl_patch_id, dtype=np.int64)
    valid_patch = (patch_id >= 0) & (patch_id < len(wall_patches))
    safe_patch = np.clip(patch_id, 0, max(len(wall_patches) - 1, 0))
    patch = wall_patches[safe_patch].astype(np.float32)
    patch[~valid_patch] = 0.0

    # Wall-patch geometry for the top candidates.  These are continuous
    # official point-cloud quantities, not mesh labels.
    patch_n = patch[..., :2]
    patch_c = patch[..., 2]
    patch_len = np.maximum(patch[..., 4] - patch[..., 3], 0.0)
    patch_h = np.maximum(patch[..., 6] - patch[..., 5], 0.0)
    patch_count = np.log1p(np.maximum(patch[..., 7], 0.0))
    patch_res = patch[..., 9]
    top_point = point[:, 0, :]
    top_dir = top_point - BS_POS[None, :].astype(np.float32)
    top_dir = top_dir / np.maximum(np.linalg.norm(top_dir, axis=1, keepdims=True), 1e-6)
    ue_dir = pos - top_point
    ue_dir = ue_dir / np.maximum(np.linalg.norm(ue_dir, axis=1, keepdims=True), 1e-6)
    top_normal = np.concatenate([patch_n[:, 0, :], np.zeros((len(pos), 1), dtype=np.float32)], axis=1)
    top_normal = top_normal / np.maximum(np.linalg.norm(top_normal, axis=1, keepdims=True), 1e-6)
    spec_balance = np.abs(np.sum(top_dir * top_normal, axis=1) + np.sum(ue_dir * top_normal, axis=1))
    refl_summary = np.column_stack(
        [
            score[:, 0],
            score[:, :3].sum(axis=1),
            score.mean(axis=1),
            score.max(axis=1) - np.partition(score, -2, axis=1)[:, -2],
            np.log1p(excess[:, 0]) / 7.0,
            np.log1p(np.maximum(excess[:, :3], 0.0)).mean(axis=1) / 7.0,
            path[:, 0] / 260.0,
            path[:, :3].mean(axis=1) / 260.0,
            patch_len[:, 0] / 80.0,
            patch_h[:, 0] / 35.0,
            patch_count[:, 0] / 9.0,
            patch_res[:, 0],
            spec_balance,
            top_dir[:, 0],
            top_dir[:, 1],
            top_dir[:, 2],
            ue_dir[:, 0],
            ue_dir[:, 1],
            ue_dir[:, 2],
        ]
    ).astype(np.float32)
    refl_flat = np.concatenate(
        [
            score[:, :8],
            np.log1p(excess[:, :8]) / 7.0,
            path[:, :8] / 260.0,
            patch_len[:, :8] / 80.0,
            patch_h[:, :8] / 35.0,
            patch_count[:, :8] / 9.0,
            patch_res[:, :8],
            patch_n[:, :8, :].reshape(len(pos), -1),
        ],
        axis=1,
    ).astype(np.float32)

    geom = np.column_stack(
        [
            (pos[:, 0] - 120.0) / 120.0,
            (pos[:, 1] + 80.0) / 120.0,
            pos[:, 2] / 30.0,
            dist / 250.0,
            np.log1p(dist) / 6.0,
            np.sin(az),
            np.cos(az),
            np.sin(el),
            np.cos(el),
            blocked / 160.0,
            np.log1p(np.maximum(blocked, 0.0)) / 6.0,
            refl_summary,
        ]
    ).astype(np.float32)
    out = np.concatenate([geom, reg_oh, f_raw, f_log, refl_summary, refl_flat], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


def spatial_embedding(pos: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    rel = pos - BS_POS[None, :]
    dist = np.linalg.norm(rel, axis=1)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    el = np.arctan2(rel[:, 2], np.maximum(np.hypot(rel[:, 0], rel[:, 1]), 1e-6))
    reg = regime_of(blocked)
    reg_oh = np.stack([(reg == r).astype(np.float32) for r in REGIMES], axis=1)
    feat = np.column_stack(
        [
            (pos[:, 0] - 120.0) / 120.0,
            (pos[:, 1] + 80.0) / 120.0,
            pos[:, 2] / 30.0,
            dist / 250.0,
            np.log1p(dist) / 6.0,
            np.sin(az),
            np.cos(az),
            np.sin(el),
            np.cos(el),
            blocked / 160.0,
            np.log1p(np.maximum(blocked, 0.0)) / 6.0,
        ]
    ).astype(np.float32)
    return np.nan_to_num(np.concatenate([feat, reg_oh], axis=1), nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


def gate_features(
    pos: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
    candidates: list[np.ndarray],
    current: np.ndarray,
    confs: list[np.ndarray],
    entropies: list[np.ndarray],
) -> np.ndarray:
    rel = pos - BS_POS[None, :]
    dist = np.linalg.norm(rel, axis=1)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    el = np.arctan2(rel[:, 2], np.maximum(np.hypot(rel[:, 0], rel[:, 1]), 1e-6))
    reg_oh = np.stack([(regimes == r).astype(np.float32) for r in REGIMES], axis=1)
    base = np.column_stack(
        [
            (pos[:, 0] - 120.0) / 120.0,
            (pos[:, 1] + 80.0) / 120.0,
            dist / 250.0,
            np.log1p(dist) / 6.0,
            np.sin(az),
            np.cos(az),
            np.sin(el),
            np.cos(el),
            blocked / 160.0,
            np.log1p(np.maximum(blocked, 0.0)) / 6.0,
        ]
    ).astype(np.float32)
    cos_cur = [row_cos(c, current).astype(np.float32) for c in candidates[:-1]]
    pair_cos = [
        row_cos(candidates[0], candidates[1]).astype(np.float32),
        row_cos(candidates[0], candidates[2]).astype(np.float32),
        row_cos(candidates[1], candidates[2]).astype(np.float32),
        row_cos(candidates[0], candidates[3]).astype(np.float32),
        row_cos(candidates[2], candidates[3]).astype(np.float32),
    ]
    agreement = np.minimum.reduce([np.asarray(x) for x in pair_cos[:3]]).astype(np.float32)
    divergence = (1.0 - np.minimum.reduce([np.asarray(x) for x in cos_cur[:4]])).astype(np.float32)
    feat = np.column_stack(
        [
            base,
            reg_oh,
            *confs,
            *entropies,
            *cos_cur,
            *pair_cos,
            agreement,
            divergence,
        ]
    )
    return np.nan_to_num(feat, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--scene-cache", default=ASSETS + "/official_pointcloud_scene_cache.npz")
    ap.add_argument("--base", default=ASSETS + "/cache_v112/v112_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=ASSETS + "/cache_v115/v115_Round1_Test_Channel.npy")
    ap.add_argument("--truth-support", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--train-current-support", default=ASSETS + "/train_v98like_loo_joint_target.npy")
    ap.add_argument("--current-target", default=ASSETS + "/cache_v112/v112_kernel_moe_path_support_target.npy")
    ap.add_argument("--v111-anchor", default=ASSETS + "/cache_v111/v111_manual_kernel_path_support_target.npy")
    ap.add_argument("--v109-anchor", default=ASSETS + "/cache_v109/v109_accelerated_momentum_path_support_target.npy")
    ap.add_argument("--target-out", default=ASSETS + "/cache_v115/v115_pointcloud_scene_support_target.npy")
    ap.add_argument("--weights-out", default=ASSETS + "/cache_v115/v115_pointcloud_scene_support_weights.npy")
    ap.add_argument("--moe-out", default=ASSETS + "/cache_v115/v115_pointcloud_scene_support_moe_weights.npy")
    ap.add_argument("--report", default=ASSETS + "/v115_pointcloud_scene_support_generate.json")
    ap.add_argument("--zip-out", default=str(PATHS.round1_versions / "v115_pointcloud_scene_support_candidate_pending.zip"))
    ap.add_argument("--candidate-zip", default=str(PATHS.round1_versions / "Round1_Submission_candidate_v115_pointcloud_scene_support.zip"))
    ap.add_argument("--k-cloud-strict", type=int, default=9)
    ap.add_argument("--k-cloud-broad", type=int, default=36)
    ap.add_argument("--k-spatial", type=int, default=24)
    ap.add_argument("--k-test", type=int, default=10)
    ap.add_argument("--cloud-strict-power", type=float, default=3.65)
    ap.add_argument("--cloud-broad-power", type=float, default=2.20)
    ap.add_argument("--spatial-power", type=float, default=2.35)
    ap.add_argument("--test-power", type=float, default=3.00)
    ap.add_argument("--cloud-strict-floor", type=float, default=0.40)
    ap.add_argument("--cloud-broad-floor", type=float, default=0.85)
    ap.add_argument("--spatial-floor", type=float, default=0.70)
    ap.add_argument("--test-floor", type=float, default=0.52)
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--project-batch", type=int, default=10)
    ap.add_argument("--eta", default="clear:0.0005,light:0.0035,medium:0.560,heavy:0.760")
    ap.add_argument("--min-gate", type=float, default=0.24)
    ap.add_argument("--max-gate", type=float, default=2.45)
    ap.add_argument("--conf-center", type=float, default=0.22)
    ap.add_argument("--conf-width", type=float, default=0.16)
    ap.add_argument("--entropy-low", type=float, default=0.18)
    ap.add_argument("--entropy-high", type=float, default=0.90)
    ap.add_argument("--gain-clip", default="0.40,3.30")
    ap.add_argument("--top-gain-clip", default="0.55,2.35")
    ap.add_argument("--top-components", type=int, default=1792)
    ap.add_argument("--moe-label-temp", type=float, default=0.038)
    ap.add_argument("--moe-pred-temp", type=float, default=0.88)
    ap.add_argument("--moe-ridge", type=float, default=20.0)
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
    scene_train = scene["train_features"].astype(np.float32)
    scene_test = scene["test_features"].astype(np.float32)
    wall_patches = scene["wall_patches"].astype(np.float32)
    train_refl_score = scene["train_reflection_score"].astype(np.float32)
    train_refl_path = scene["train_reflection_path_length"].astype(np.float32)
    train_refl_excess = scene["train_reflection_excess_length"].astype(np.float32)
    train_refl_point = scene["train_reflection_point"].astype(np.float32)
    train_refl_patch = scene["train_reflection_patch_id"].astype(np.int64)
    test_refl_score = scene["test_reflection_score"].astype(np.float32)
    test_refl_path = scene["test_reflection_path_length"].astype(np.float32)
    test_refl_excess = scene["test_reflection_excess_length"].astype(np.float32)
    test_refl_point = scene["test_reflection_point"].astype(np.float32)
    test_refl_patch = scene["test_reflection_patch_id"].astype(np.int64)

    print("load supports", flush=True)
    truth = normalize_support(np.load(args.truth_support, mmap_mode="r"), float(args.floor))
    train_current = normalize_support(np.load(args.train_current_support, mmap_mode="r"), float(args.floor))
    current = normalize_support(np.load(args.current_target), float(args.floor))
    v111_anchor = normalize_support(np.load(args.v111_anchor), float(args.floor))
    v109_anchor = normalize_support(np.load(args.v109_anchor), float(args.floor))

    print("build official point-cloud scene embeddings", flush=True)
    cloud_train, cloud_test = standardize(
        official_scene_embedding(
            pos_train,
            blocked_train,
            scene_train,
            train_refl_score,
            train_refl_path,
            train_refl_excess,
            train_refl_point,
            train_refl_patch,
            wall_patches,
        ),
        official_scene_embedding(
            pos_test,
            blocked_test,
            scene_test,
            test_refl_score,
            test_refl_path,
            test_refl_excess,
            test_refl_point,
            test_refl_patch,
            wall_patches,
        ),
    )
    spatial_train, spatial_test = standardize(
        spatial_embedding(pos_train, blocked_train),
        spatial_embedding(pos_test, blocked_test),
    )

    print("knn: official cloud strict/broad, spatial, test smooth", flush=True)
    idx_cloud_strict_tr, dist_cloud_strict_tr = pairwise_knn(cloud_train, cloud_train, int(args.k_cloud_strict), exclude_self=True)
    idx_cloud_broad_tr, dist_cloud_broad_tr = pairwise_knn(cloud_train, cloud_train, int(args.k_cloud_broad), exclude_self=True)
    idx_spatial_tr, dist_spatial_tr = pairwise_knn(spatial_train, spatial_train, int(args.k_spatial), exclude_self=True)
    idx_smooth_tr, dist_smooth_tr = pairwise_knn(cloud_train, cloud_train, int(args.k_test), exclude_self=True)

    idx_cloud_strict_te, dist_cloud_strict_te = pairwise_knn(cloud_test, cloud_train, int(args.k_cloud_strict), exclude_self=False)
    idx_cloud_broad_te, dist_cloud_broad_te = pairwise_knn(cloud_test, cloud_train, int(args.k_cloud_broad), exclude_self=False)
    idx_spatial_te, dist_spatial_te = pairwise_knn(spatial_test, spatial_train, int(args.k_spatial), exclude_self=False)
    idx_test_te, dist_test_te = pairwise_knn(cloud_test, cloud_test, int(args.k_test) + 1, exclude_self=True)
    idx_test_te = idx_test_te[:, : int(args.k_test)]
    dist_test_te = dist_test_te[:, : int(args.k_test)]

    w_cloud_strict_tr = weights_from_dist(dist_cloud_strict_tr, float(args.cloud_strict_power), float(args.cloud_strict_floor))
    w_cloud_broad_tr = weights_from_dist(dist_cloud_broad_tr, float(args.cloud_broad_power), float(args.cloud_broad_floor))
    w_spatial_tr = weights_from_dist(dist_spatial_tr, float(args.spatial_power), float(args.spatial_floor))
    w_smooth_tr = weights_from_dist(dist_smooth_tr, float(args.test_power), float(args.test_floor))

    w_cloud_strict_te = weights_from_dist(dist_cloud_strict_te, float(args.cloud_strict_power), float(args.cloud_strict_floor))
    w_cloud_broad_te = weights_from_dist(dist_cloud_broad_te, float(args.cloud_broad_power), float(args.cloud_broad_floor))
    w_spatial_te = weights_from_dist(dist_spatial_te, float(args.spatial_power), float(args.spatial_floor))
    w_test_te = weights_from_dist(dist_test_te, float(args.test_power), float(args.test_floor))

    print("build train LOO official expert targets", flush=True)
    tr_cloud_strict = log_average(truth, idx_cloud_strict_tr, w_cloud_strict_tr, float(args.floor), int(args.batch))
    tr_cloud_broad = log_average(truth, idx_cloud_broad_tr, w_cloud_broad_tr, float(args.floor), int(args.batch))
    tr_spatial = log_average(truth, idx_spatial_tr, w_spatial_tr, float(args.floor), int(args.batch))
    tr_smooth = log_average(train_current, idx_smooth_tr, w_smooth_tr, float(args.floor), int(args.batch))
    tr_current = train_current

    print("build test official expert targets", flush=True)
    te_cloud_strict = log_average(truth, idx_cloud_strict_te, w_cloud_strict_te, float(args.floor), int(args.batch))
    te_cloud_broad = log_average(truth, idx_cloud_broad_te, w_cloud_broad_te, float(args.floor), int(args.batch))
    te_spatial = log_average(truth, idx_spatial_te, w_spatial_te, float(args.floor), int(args.batch))
    te_smooth = log_average(current, idx_test_te, w_test_te, float(args.floor), int(args.batch))

    train_candidates = [tr_cloud_strict, tr_cloud_broad, tr_spatial, tr_smooth, tr_current]
    test_candidates = [te_cloud_strict, te_cloud_broad, te_spatial, te_smooth, current]
    expert_names = ["cloud_strict", "cloud_broad", "spatial", "cloud_smooth_current", "current"]
    expert_cos = np.stack([row_cos(c, truth) for c in train_candidates], axis=1).astype(np.float32)

    train_confs = [kernel_conf(w_cloud_strict_tr, 0.32), kernel_conf(w_cloud_broad_tr, 0.24), kernel_conf(w_spatial_tr, 0.26), kernel_conf(w_smooth_tr, 0.40)]
    test_confs = [kernel_conf(w_cloud_strict_te, 0.32), kernel_conf(w_cloud_broad_te, 0.24), kernel_conf(w_spatial_te, 0.26), kernel_conf(w_test_te, 0.40)]
    train_ents = [entropy01(w_cloud_strict_tr), entropy01(w_cloud_broad_tr), entropy01(w_spatial_tr), entropy01(w_smooth_tr)]
    test_ents = [entropy01(w_cloud_strict_te), entropy01(w_cloud_broad_te), entropy01(w_spatial_te), entropy01(w_test_te)]
    train_feat = gate_features(pos_train, blocked_train, regimes_train, train_candidates, tr_current, train_confs, train_ents)
    test_feat = gate_features(pos_test, blocked_test, regimes_test, test_candidates, current, test_confs, test_ents)
    train_prob, test_prob, moe_report = fit_moe_gate(
        train_feat,
        test_feat,
        expert_cos,
        float(args.moe_label_temp),
        float(args.moe_pred_temp),
        float(args.moe_ridge),
    )
    moe_target = log_blend(test_candidates, test_prob, float(args.floor), int(args.batch))

    # Point-cloud-scene target is still less exact than original Sionna
    # path-token, so keep a current anchor and let geometry consensus decide
    # how much to trust it.
    cloud_agree = np.minimum.reduce(
        [
            row_cos(te_cloud_strict, te_cloud_broad),
            row_cos(te_cloud_strict, te_spatial),
            row_cos(te_cloud_broad, te_spatial),
        ]
    ).astype(np.float32)
    cloud_div = (1.0 - np.minimum.reduce([row_cos(te_cloud_strict, current), row_cos(te_cloud_broad, current), row_cos(te_smooth, current)])).astype(np.float32)
    moe_conf = np.clip(1.0 - test_prob[:, -1], 0.0, 1.0)
    shape_conf = np.clip((row_cos(moe_target, current) - 0.88) / 0.10, 0.0, 1.0)
    reg_strength = {"clear": 0.020, "light": 0.055, "medium": 0.540, "heavy": 0.620}
    strength = np.array([reg_strength[str(r)] for r in regimes_test], dtype=np.float32)
    strength *= (0.50 + 0.50 * np.clip((cloud_agree - 0.55) / 0.28, 0.0, 1.0)).astype(np.float32)
    strength *= (0.70 + 0.30 * np.clip(cloud_div / 0.14, 0.0, 1.0)).astype(np.float32)
    strength *= (0.62 + 0.38 * moe_conf).astype(np.float32)
    strength *= (0.60 + 0.40 * shape_conf).astype(np.float32)
    strength = np.clip(strength, 0.0, 0.68).astype(np.float32)
    final_coeff = np.zeros((len(current), 4), dtype=np.float32)
    final_coeff[:, 0] = 1.0 - strength
    final_coeff[:, 1] = 0.92 * strength
    final_coeff[:, 2] = 0.055 * strength
    final_coeff[:, 3] = 0.025 * strength
    target = log_blend([current, moe_target, v111_anchor, v109_anchor], final_coeff, float(args.floor), int(args.batch))

    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.target_out, target)
    np.save(args.weights_out, w_cloud_strict_te.astype(np.float32))
    np.save(args.moe_out, test_prob.astype(np.float32))

    proj = projection(target, w_cloud_strict_te.astype(np.float32), regimes_test, pos_test, args)
    payload = {
        "method": "v115_pointcloud_scene_support",
        "base": args.base,
        "operation": "official point-cloud scene target on top of v112; uses reconstructed wall patches/corridor/specular candidates; no Sionna path-token/mesh/object/material labels; phase/scalar/total power preserved",
        "args": vars(args),
        "experts": expert_names,
        "moe": moe_report,
        "train_expert_cos": {name: summarize(expert_cos[:, i]) for i, name in enumerate(expert_names)},
        "test_moe_weight": {name: summarize(test_prob[:, i]) for i, name in enumerate(expert_names)},
        "graph": {
            "cloud_strict_weight_max": summarize(w_cloud_strict_te.max(axis=1)),
            "cloud_broad_weight_max": summarize(w_cloud_broad_te.max(axis=1)),
            "spatial_weight_max": summarize(w_spatial_te.max(axis=1)),
            "test_weight_max": summarize(w_test_te.max(axis=1)),
            "cloud_strict_entropy": summarize(entropy01(w_cloud_strict_te)),
            "cloud_broad_entropy": summarize(entropy01(w_cloud_broad_te)),
            "spatial_entropy": summarize(entropy01(w_spatial_te)),
            "test_entropy": summarize(entropy01(w_test_te)),
            "cloud_agreement": summarize(cloud_agree),
            "cloud_divergence": summarize(cloud_div),
            "moe_strength": summarize(strength),
            "moe_conf_noncurrent": summarize(moe_conf),
            "moe_shape_conf": summarize(shape_conf),
        },
        "target_cos": {
            "target_vs_current": summarize(row_cos(target, current)),
            "moe_vs_current": summarize(row_cos(moe_target, current)),
            "target_vs_moe": summarize(row_cos(target, moe_target)),
            "target_vs_v111": summarize(row_cos(target, v111_anchor)),
            "target_vs_v109": summarize(row_cos(target, v109_anchor)),
            "cloud_strict_vs_current": summarize(row_cos(te_cloud_strict, current)),
            "cloud_broad_vs_current": summarize(row_cos(te_cloud_broad, current)),
            "spatial_vs_current": summarize(row_cos(te_spatial, current)),
            "smooth_vs_current": summarize(row_cos(te_smooth, current)),
            "strict_broad_agreement": summarize(row_cos(te_cloud_strict, te_cloud_broad)),
            "strict_spatial_agreement": summarize(row_cos(te_cloud_strict, te_spatial)),
            "broad_spatial_agreement": summarize(row_cos(te_cloud_broad, te_spatial)),
        },
        "projection": proj,
        "outputs": {"submission_npy": args.output, "target": args.target_out, "weights": args.weights_out, "moe": args.moe_out},
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    def write_zip(path: str) -> None:
        files = [
            (args.output, "Round1_Test_Channel.npy"),
            (__file__, "generate_v115_pointcloud_scene_support.py"),
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
