#!/usr/bin/env python3
"""Generate v130: family-conditioned KNN bad-point support residual.

v128/v129 showed that train bad-point residual support patches transfer, but
the global residual-template selector is nearly saturated.  v130 switches the
information slice: it assigns each point to a coarse geometry/visibility family
and directly averages residual ratios from nearby bad training points in that
family.  This is intentionally more aggressive than template clustering.

The output still changes only collapsed H/V/delay support.  Total power, AD
phase, and pol/UE allocation are inherited from the v129 base.
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
    log_blend,
    normalize_support,
    pairwise_knn,
    row_cos,
    standardize,
    weights_from_dist,
)
from generate_v115_pointcloud_scene_support import official_scene_embedding, spatial_embedding
from generate_v121_support_template_manifold import support_signature
from generate_v128_badpoint_support_residual_templates import build_bad_mask, collapsed_support_from_ad
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
        out[key.strip()] = float(value)
    return out


def scene_columns(scene: np.lib.npyio.NpzFile, prefix: str) -> dict[str, np.ndarray]:
    features = scene[f"{prefix}_features"].astype(np.float32)
    names = [str(x) for x in scene["feature_names"]]
    idx = {name: i for i, name in enumerate(names)}

    def col(name: str, default: float = 0.0) -> np.ndarray:
        if name not in idx:
            return np.full(features.shape[0], default, dtype=np.float32)
        return features[:, idx[name]].astype(np.float32)

    score = scene[f"{prefix}_reflection_score"].astype(np.float32)
    excess = scene[f"{prefix}_reflection_excess_length"].astype(np.float32)
    top_id = np.argmax(score, axis=1)
    rows = np.arange(score.shape[0])
    return {
        "los_frac": col("pc_los_height_block_frac"),
        "los_count": col("pc_los_height_block_count"),
        "near_bs": col("pc_los_near_bs_block_frac"),
        "near_ue": col("pc_los_near_ue_block_frac"),
        "local_min": col("pc_local_ray_min", 80.0),
        "local_p10": col("pc_local_ray_p10", 80.0),
        "nn_min": col("pc_los_nn_min_dist", 8.0),
        "nn_inv": col("pc_los_nn_mean_inv"),
        "top_ref_score": score.max(axis=1).astype(np.float32),
        "top_ref_excess": excess[rows, top_id].astype(np.float32),
    }


def family_labels(
    blocked: np.ndarray,
    scene_cols: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    regimes = regime_of(blocked)
    los_frac = scene_cols["los_frac"]
    near_bs = scene_cols["near_bs"]
    near_ue = scene_cols["near_ue"]
    local_min = scene_cols["local_min"]
    top_ref = scene_cols["top_ref_score"]
    excess = scene_cols["top_ref_excess"]

    fam = np.zeros(len(blocked), dtype=np.int64)
    # 0 clear/LOS-like
    fam[(regimes == "clear") & (los_frac < 0.02)] = 0
    # 1 light or local wall-near, where existing support is unstable but not deep NLOS.
    fam[((regimes == "light") | ((local_min < 6.0) & (blocked < 30))) & (fam == 0)] = 1
    # 2 medium with plausible single-reflection cue.
    fam[(regimes == "medium") & (top_ref > 0.50) & (excess < 160.0)] = 2
    # 3 medium diffuse/weak-reflection.
    fam[(regimes == "medium") & (fam == 0)] = 3
    # 4 heavy with strong raw reflection and moderate excess.
    fam[(regimes == "heavy") & (top_ref > 0.55) & (excess < 220.0)] = 4
    # 5 heavy near-obstacle / near-end blockage.
    fam[(regimes == "heavy") & ((near_bs > 0.18) | (near_ue > 0.18) | (local_min < 4.0))] = 5
    # 6 heavy deep blocked with no convincing reflection cue.
    fam[(regimes == "heavy") & (fam == 0)] = 6
    # 7 high LOS blockage but not heavy: transition/edge cases.
    fam[(regimes != "heavy") & (los_frac > 0.18) & (fam <= 1)] = 7

    family_features = np.column_stack(
        [
            fam / 7.0,
            los_frac,
            np.log1p(np.maximum(scene_cols["los_count"], 0.0)) / 5.0,
            near_bs,
            near_ue,
            np.clip(local_min, 0.0, 80.0) / 80.0,
            np.clip(scene_cols["local_p10"], 0.0, 80.0) / 80.0,
            np.clip(scene_cols["nn_min"], 0.0, 10.0) / 10.0,
            np.log1p(np.maximum(scene_cols["nn_inv"], 0.0)) / 5.0,
            top_ref,
            np.log1p(np.maximum(excess, 0.0)) / 6.0,
        ]
    ).astype(np.float32)
    return fam, np.nan_to_num(family_features, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


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
    family_train_feat: np.ndarray,
    family_test_feat: np.ndarray,
    family_train: np.ndarray,
    family_test: np.ndarray,
    floor: float,
) -> tuple[np.ndarray, np.ndarray]:
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
    fam_aux_train, fam_aux_test = standardize(family_train_feat, family_test_feat)
    prob_train, prob_test = standardize(train_template_prob.astype(np.float32), test_template_prob.astype(np.float32))

    one_train = np.eye(8, dtype=np.float32)[family_train]
    one_test = np.eye(8, dtype=np.float32)[family_test]
    feat_train = np.concatenate(
        [
            0.66 * cloud_train,
            0.42 * spatial_train,
            0.66 * sig_train,
            0.34 * prob_train,
            1.35 * fam_aux_train,
            2.20 * one_train,
        ],
        axis=1,
    ).astype(np.float32)
    feat_test = np.concatenate(
        [
            0.66 * cloud_test,
            0.42 * spatial_test,
            0.66 * sig_test,
            0.34 * prob_test,
            1.35 * fam_aux_test,
            2.20 * one_test,
        ],
        axis=1,
    ).astype(np.float32)
    return feat_train, feat_test


def weighted_log_ratio_target(
    current: np.ndarray,
    bad_log_ratio: np.ndarray,
    idx: np.ndarray,
    weights: np.ndarray,
    floor: float,
    batch: int,
) -> np.ndarray:
    out = np.empty_like(current, dtype=np.float32)
    flat_shape = (MH, MV, S_SC)
    bad_flat = bad_log_ratio.reshape(bad_log_ratio.shape[0], -1).astype(np.float32)
    for start in range(0, current.shape[0], int(batch)):
        end = min(start + int(batch), current.shape[0])
        lr = np.einsum("bk,bkf->bf", weights[start:end].astype(np.float32), bad_flat[idx[start:end]])
        ratio = np.exp(np.clip(lr, math.log(0.12), math.log(8.50))).reshape(end - start, *flat_shape)
        target = current[start:end].astype(np.float32) * ratio.astype(np.float32)
        target = np.maximum(target, floor).astype(np.float32)
        target /= np.maximum(target.sum(axis=(1, 2, 3), keepdims=True), TINY)
        out[start:end] = target.astype(np.float32)
        print(f"family residual target {end}/{current.shape[0]}", flush=True)
    return out


def sharpen_weights(weights: np.ndarray, keep: int, power: float) -> np.ndarray:
    keep = min(int(keep), weights.shape[1])
    w = weights[:, :keep].astype(np.float32) ** float(power)
    w /= np.maximum(w.sum(axis=1, keepdims=True), TINY)
    return w.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--scene-cache", default=ASSETS + "/official_pointcloud_scene_cache.npz")
    ap.add_argument("--base", default=ASSETS + "/cache_v129/v129_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=ASSETS + "/cache_v130/v130_Round1_Test_Channel.npy")
    ap.add_argument("--truth-support", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--train-base-ad", default=ASSETS + "/train_v125eq_crossfit_base_ad.npy")
    ap.add_argument("--train-base-support-out", default=ASSETS + "/cache_v128/train_v125eq_collapsed_support.npy")
    ap.add_argument("--current-target", default=ASSETS + "/cache_v129/v129_apex_badpoint_support_residual_target.npy")
    ap.add_argument("--v128-target", default=ASSETS + "/cache_v128/v128_badpoint_support_residual_target.npy")
    ap.add_argument("--v124-target", default=ASSETS + "/cache_v124/v124_visibility_misleading_overdrive_target.npy")
    ap.add_argument("--train-template-probs", default=ASSETS + "/train_v125eq_crossfit_template_probs.npy")
    ap.add_argument("--test-template-probs", default=ASSETS + "/cache_v124/v124_visibility_misleading_overdrive_probs.npy")
    ap.add_argument("--target-out", default=ASSETS + "/cache_v130/v130_family_knn_badpoint_residual_target.npy")
    ap.add_argument("--weights-out", default=ASSETS + "/cache_v130/v130_family_knn_badpoint_residual_weights.npy")
    ap.add_argument("--report", default=ASSETS + "/v130_family_knn_badpoint_residual_generate.json")
    ap.add_argument("--zip-out", default=str(PATHS.round1_versions / "v130_family_knn_badpoint_residual_candidate_pending.zip"))
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--ad-batch", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project-batch", type=int, default=10)
    ap.add_argument("--bad-quantile", type=float, default=0.48)
    ap.add_argument("--min-bad-cos", type=float, default=0.982)
    ap.add_argument("--k-bad", type=int, default=58)
    ap.add_argument("--bad-power", type=float, default=3.85)
    ap.add_argument("--bad-floor", type=float, default=0.26)
    ap.add_argument("--weight-keep", type=int, default=10)
    ap.add_argument("--weight-power", type=float, default=2.1)
    ap.add_argument("--eta", default="clear:0.0001,light:0.012,medium:1.050,heavy:1.560")
    ap.add_argument("--min-gate", type=float, default=0.38)
    ap.add_argument("--max-gate", type=float, default=4.80)
    ap.add_argument("--conf-center", type=float, default=0.10)
    ap.add_argument("--conf-width", type=float, default=0.15)
    ap.add_argument("--entropy-low", type=float, default=0.10)
    ap.add_argument("--entropy-high", type=float, default=0.98)
    ap.add_argument("--gain-clip", default="0.22,5.80")
    ap.add_argument("--top-gain-clip", default="0.30,4.80")
    ap.add_argument("--top-components", type=int, default=4608)
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

    truth = normalize_support(np.load(args.truth_support, mmap_mode="r"), float(args.floor))
    train_base = collapsed_support_from_ad(args.train_base_ad, args.train_base_support_out, float(args.floor), int(args.ad_batch))
    current = normalize_support(np.load(args.current_target), float(args.floor))
    v128 = normalize_support(np.load(args.v128_target), float(args.floor))
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
    cols_train = scene_columns(scene, "train")
    cols_test = scene_columns(scene, "test")
    fam_train, fam_train_feat = family_labels(blocked_train, cols_train)
    fam_test, fam_test_feat = family_labels(blocked_test, cols_test)

    feat_train, feat_test = build_features(
        pos_train,
        pos_test,
        blocked_train,
        blocked_test,
        train_base,
        current,
        train_template_prob,
        test_template_prob,
        scene,
        fam_train_feat,
        fam_test_feat,
        fam_train,
        fam_test,
        float(args.floor),
    )
    bad_feat = feat_train[bad_mask]
    idx, dist = pairwise_knn(feat_test, bad_feat, int(args.k_bad), exclude_self=False)
    weights = weights_from_dist(dist, float(args.bad_power), float(args.bad_floor))
    bad_ids = np.flatnonzero(bad_mask)
    bad_fam = fam_train[bad_mask]
    same_family = bad_fam[idx] == fam_test[:, None]
    family_ratio = same_family.mean(axis=1).astype(np.float32)
    # Strongly prefer same-family bad points, but keep a fallback so rare test
    # families do not become unstable.
    weights = weights * (0.18 + 1.82 * same_family.astype(np.float32))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)

    ratio_clip = (0.12, 8.50)
    log_ratio = np.log(np.maximum(truth[bad_mask], float(args.floor))) - np.log(
        np.maximum(train_base[bad_mask], float(args.floor))
    )
    log_ratio = np.clip(log_ratio, math.log(ratio_clip[0]), math.log(ratio_clip[1])).astype(np.float32)
    residual_target = weighted_log_ratio_target(current, log_ratio, idx, weights, float(args.floor), int(args.batch))

    avg_badness = (weights * (1.0 - train_cos[bad_ids][idx])).sum(axis=1).astype(np.float32)
    resid_cos = row_cos(residual_target, current).astype(np.float32)
    strength = np.clip((avg_badness - 0.16) / 0.34, 0.0, 1.0)
    strength *= (0.50 + 0.50 * np.clip((family_ratio - 0.35) / 0.45, 0.0, 1.0)).astype(np.float32)
    strength *= (0.82 + 0.18 * np.clip((1.0 - resid_cos) / 0.035, 0.0, 1.0)).astype(np.float32)
    family_scale = np.array([0.03, 0.18, 0.82, 0.70, 1.35, 1.15, 1.45, 0.52], dtype=np.float32)
    strength = np.clip(strength * family_scale[fam_test], 0.0, 1.0).astype(np.float32)

    coeff = np.zeros((len(current), 4), dtype=np.float32)
    coeff[:, 0] = 1.0 - strength
    coeff[:, 1] = 0.92 * strength
    coeff[:, 2] = 0.055 * strength
    coeff[:, 3] = 0.025 * strength
    target = log_blend([current, residual_target, v128, v124], coeff, float(args.floor), int(args.batch))

    weights_proj = sharpen_weights(weights, int(args.weight_keep), float(args.weight_power))
    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.target_out, target.astype(np.float32))
    np.save(args.weights_out, weights_proj.astype(np.float32))

    proj = projection(target, weights_proj, regimes_test, pos_test, args)
    payload = {
        "method": "v130_family_knn_badpoint_residual",
        "base": args.base,
        "operation": "family-conditioned KNN average of train bad-point support residual ratios; support-only projection preserving v129 total power/phase/pol-UE allocation",
        "args": vars(args),
        "bad_train": {
            "count": int(bad_mask.sum()),
            "fraction": float(np.mean(bad_mask)),
            "train_base_vs_truth_cos": summarize(train_cos),
            "bad_cos": summarize(train_cos[bad_mask]),
            "bad_regime_counts": {r: int(np.sum(regimes_train[bad_mask] == r)) for r in REGIMES},
            "bad_family_counts": {str(i): int(np.sum(fam_train[bad_mask] == i)) for i in range(8)},
        },
        "test_family_counts": {str(i): int(np.sum(fam_test == i)) for i in range(8)},
        "selector": {
            "family_ratio": summarize(family_ratio),
            "avg_badness": summarize(avg_badness),
            "residual_target_vs_current": summarize(resid_cos),
            "strength": summarize(strength),
            "nearest_bad_dist": summarize(np.sqrt(np.maximum(dist[:, 0], 0.0))),
        },
        "target_cos": {
            "target_vs_current": summarize(row_cos(target, current)),
            "target_vs_residual": summarize(row_cos(target, residual_target)),
            "target_vs_v128": summarize(row_cos(target, v128)),
            "target_vs_v124": summarize(row_cos(target, v124)),
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

    files = [
        (args.output, "Round1_Test_Channel.npy"),
        (__file__, "generate_v130_family_knn_badpoint_residual.py"),
        (ASSETS + "/generate_v128_badpoint_support_residual_templates.py", "generate_v128_badpoint_support_residual_templates.py"),
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
