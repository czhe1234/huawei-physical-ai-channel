#!/usr/bin/env python3
"""Generate v122: multiscale support-template manifold apex.

v121 confirmed that train-truth support templates contain useful online signal.
v122 makes that idea more aggressive in the most defensible way:

* base is current online best v121;
* global support templates use several granularities instead of one cluster count;
* a regime-specific template branch is added for clear/light/medium/heavy;
* a test-test smoothing branch enforces local consistency among test points;
* phase, complex scalar structure, and per-point total power are preserved.

This is intentionally a high-risk/high-upside candidate, but it does not reuse
the v118/v120 inverse-RT direct path template that online feedback rejected.
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


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def make_template_target(
    truth: np.ndarray,
    train_current: np.ndarray,
    current: np.ndarray,
    scene_train: np.ndarray,
    scene_test: np.ndarray,
    geom_train: np.ndarray,
    geom_test: np.ndarray,
    clusters: int,
    svd_components: int,
    seed: int,
    args: argparse.Namespace,
    train_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build one soft template target.

    If ``train_mask`` is given, the template/probability bank is fit only on
    that subset.  This is used for regime-specific templates.
    """
    if train_mask is None:
        tr_ids = np.arange(len(truth), dtype=np.int64)
    else:
        tr_ids = np.where(train_mask)[0].astype(np.int64)
    if len(tr_ids) < 12:
        tr_ids = np.arange(len(truth), dtype=np.int64)
    k = min(int(clusters), max(4, len(tr_ids) // 3))

    support_x = np.log(np.maximum(truth[tr_ids].reshape(len(tr_ids), -1), float(args.floor))).astype(np.float32)
    emb, _, evr = fit_support_svd(support_x, min(int(svd_components), max(4, len(tr_ids) - 2)), seed)
    labels_sub = cluster_embeddings(emb, k, seed)
    prototypes = build_prototypes(truth[tr_ids], labels_sub, k, float(args.floor))
    labels_full = np.zeros(len(truth), dtype=np.int64)
    labels_full[tr_ids] = labels_sub

    scene_bank = scene_train[tr_ids]
    geom_bank = geom_train[tr_ids]
    idx_scene_te, dist_scene_te = pairwise_knn(scene_test, scene_bank, int(args.k_scene), exclude_self=False)
    idx_geom_te, dist_geom_te = pairwise_knn(geom_test, geom_bank, int(args.k_geom), exclude_self=False)
    w_scene_te = weights_from_dist(dist_scene_te, float(args.scene_power), float(args.scene_floor))
    w_geom_te = weights_from_dist(dist_geom_te, float(args.geom_power), float(args.geom_floor))

    eye = np.eye(k, dtype=np.float32)
    prob_scene = (w_scene_te[:, :, None] * eye[labels_sub[idx_scene_te]]).sum(axis=1)
    prob_geom = (w_geom_te[:, :, None] * eye[labels_sub[idx_geom_te]]).sum(axis=1)

    proto_flat = prototypes.reshape(k, -1).astype(np.float32)
    proto_norm = proto_flat / np.maximum(np.linalg.norm(proto_flat, axis=1, keepdims=True), TINY)
    cur_flat = current.reshape(len(current), -1).astype(np.float32)
    cur_norm = cur_flat / np.maximum(np.linalg.norm(cur_flat, axis=1, keepdims=True), TINY)
    prob_cur = softmax_np(cur_norm @ proto_norm.T, temp=float(args.proto_temp))

    sw = float(args.scene_weight)
    gw = float(args.geom_weight)
    cw = float(args.current_weight)
    prob = (sw * prob_scene + gw * prob_geom + cw * prob_cur) / max(sw + gw + cw, 1e-6)
    prob /= np.maximum(prob.sum(axis=1, keepdims=True), TINY)
    target = template_blend(prototypes, prob, float(args.floor), int(args.batch))

    # LOO-ish diagnostic on the chosen train subset, not used for generation.
    idx_scene_tr, dist_scene_tr = pairwise_knn(scene_train[tr_ids], scene_train[tr_ids], min(int(args.k_scene), len(tr_ids) - 1), exclude_self=True)
    idx_geom_tr, dist_geom_tr = pairwise_knn(geom_train[tr_ids], geom_train[tr_ids], min(int(args.k_geom), len(tr_ids) - 1), exclude_self=True)
    w_scene_tr = weights_from_dist(dist_scene_tr, float(args.scene_power), float(args.scene_floor))
    w_geom_tr = weights_from_dist(dist_geom_tr, float(args.geom_power), float(args.geom_floor))
    prob_scene_tr = (w_scene_tr[:, :, None] * eye[labels_sub[idx_scene_tr]]).sum(axis=1)
    prob_geom_tr = (w_geom_tr[:, :, None] * eye[labels_sub[idx_geom_tr]]).sum(axis=1)
    tr_flat = train_current[tr_ids].reshape(len(tr_ids), -1).astype(np.float32)
    tr_norm = tr_flat / np.maximum(np.linalg.norm(tr_flat, axis=1, keepdims=True), TINY)
    prob_cur_tr = softmax_np(tr_norm @ proto_norm.T, temp=float(args.proto_temp))
    prob_tr = (sw * prob_scene_tr + gw * prob_geom_tr + cw * prob_cur_tr) / max(sw + gw + cw, 1e-6)
    prob_tr /= np.maximum(prob_tr.sum(axis=1, keepdims=True), TINY)
    target_tr = template_blend(prototypes, prob_tr, float(args.floor), int(args.batch))
    diag = {
        "clusters": int(k),
        "train_count": int(len(tr_ids)),
        "svd_evr_sum": float(np.sum(evr)),
        "train_target_vs_truth": summarize(row_cos(target_tr, truth[tr_ids])),
        "train_current_vs_truth": summarize(row_cos(train_current[tr_ids], truth[tr_ids])),
        "prob_top1": summarize(topk_prob(prob, 8)[:, 0]),
        "prob_entropy": summarize(entropy01(prob)),
        "target_vs_current": summarize(row_cos(target, current)),
    }
    return target, prob, diag


def regime_specific_target(
    truth: np.ndarray,
    train_current: np.ndarray,
    current: np.ndarray,
    scene_train: np.ndarray,
    scene_test: np.ndarray,
    geom_train: np.ndarray,
    geom_test: np.ndarray,
    regimes_train: np.ndarray,
    regimes_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    cluster_map = {"clear": 32, "light": 20, "medium": 56, "heavy": 128}
    target = np.empty_like(current, dtype=np.float32)
    prob_top = np.zeros((len(current), 12), dtype=np.float32)
    report: dict[str, object] = {}
    for i, reg in enumerate(REGIMES):
        mask_tr = regimes_train == reg
        mask_te = regimes_test == reg
        if not np.any(mask_te):
            continue
        tgt_all, prob, diag = make_template_target(
            truth,
            train_current,
            current,
            scene_train,
            scene_test,
            geom_train,
            geom_test,
            cluster_map[reg],
            int(args.svd_components),
            int(args.seed) + 200 + i * 17,
            args,
            train_mask=mask_tr,
        )
        target[mask_te] = tgt_all[mask_te]
        prob_top[mask_te] = topk_prob(prob[mask_te], 12)
        report[reg] = diag
    return target, prob_top, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--scene-cache", default=ASSETS + "/official_pointcloud_scene_cache.npz")
    ap.add_argument("--base", default=ASSETS + "/cache_v121/v121_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=ASSETS + "/cache_v122/v122_Round1_Test_Channel.npy")
    ap.add_argument("--truth-support", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--train-current-support", default=ASSETS + "/train_v98like_loo_joint_target.npy")
    ap.add_argument("--current-target", default=ASSETS + "/cache_v121/v121_support_template_manifold_target.npy")
    ap.add_argument("--v115-anchor", default=ASSETS + "/cache_v115/v115_pointcloud_scene_support_target.npy")
    ap.add_argument("--v112-anchor", default=ASSETS + "/cache_v112/v112_kernel_moe_path_support_target.npy")
    ap.add_argument("--v111-anchor", default=ASSETS + "/cache_v111/v111_manual_kernel_path_support_target.npy")
    ap.add_argument("--target-out", default=ASSETS + "/cache_v122/v122_multiscale_template_apex_target.npy")
    ap.add_argument("--weights-out", default=ASSETS + "/cache_v122/v122_multiscale_template_apex_weights.npy")
    ap.add_argument("--report", default=ASSETS + "/v122_multiscale_template_apex_generate.json")
    ap.add_argument("--zip-out", default=str(PATHS.round1_versions / "v122_multiscale_template_apex_candidate_pending.zip"))
    ap.add_argument("--candidate-zip", default=str(PATHS.round1_versions / "Round1_Submission_candidate_v122_multiscale_template_apex.zip"))
    ap.add_argument("--cluster-list", default="64,112,192")
    ap.add_argument("--svd-components", type=int, default=80)
    ap.add_argument("--seed", type=int, default=12211)
    ap.add_argument("--k-geom", type=int, default=30)
    ap.add_argument("--k-scene", type=int, default=20)
    ap.add_argument("--k-test", type=int, default=12)
    ap.add_argument("--geom-power", type=float, default=2.70)
    ap.add_argument("--scene-power", type=float, default=3.25)
    ap.add_argument("--test-power", type=float, default=3.15)
    ap.add_argument("--geom-floor", type=float, default=0.60)
    ap.add_argument("--scene-floor", type=float, default=0.46)
    ap.add_argument("--test-floor", type=float, default=0.50)
    ap.add_argument("--proto-temp", type=float, default=0.018)
    ap.add_argument("--scene-weight", type=float, default=0.46)
    ap.add_argument("--geom-weight", type=float, default=0.22)
    ap.add_argument("--current-weight", type=float, default=0.32)
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project-batch", type=int, default=10)
    ap.add_argument("--eta", default="clear:0.0004,light:0.0048,medium:0.920,heavy:1.280")
    ap.add_argument("--min-gate", type=float, default=0.22)
    ap.add_argument("--max-gate", type=float, default=2.70)
    ap.add_argument("--conf-center", type=float, default=0.16)
    ap.add_argument("--conf-width", type=float, default=0.14)
    ap.add_argument("--entropy-low", type=float, default=0.18)
    ap.add_argument("--entropy-high", type=float, default=0.91)
    ap.add_argument("--gain-clip", default="0.34,4.00")
    ap.add_argument("--top-gain-clip", default="0.50,2.90")
    ap.add_argument("--top-components", type=int, default=3200)
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

    print("load supports", flush=True)
    truth = normalize_support(np.load(args.truth_support, mmap_mode="r"), float(args.floor))
    train_current = normalize_support(np.load(args.train_current_support, mmap_mode="r"), float(args.floor))
    current = normalize_support(np.load(args.current_target), float(args.floor))
    v115_anchor = normalize_support(np.load(args.v115_anchor), float(args.floor))
    v112_anchor = normalize_support(np.load(args.v112_anchor), float(args.floor))
    v111_anchor = normalize_support(np.load(args.v111_anchor), float(args.floor))

    print("build official scene/support features", flush=True)
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
    sig_train_raw = support_signature(train_current, float(args.floor))
    sig_test_raw = support_signature(current, float(args.floor))
    spatial_train_raw = spatial_embedding(pos_train, blocked_train)
    spatial_test_raw = spatial_embedding(pos_test, blocked_test)
    cloud_train, cloud_test = standardize(cloud_train_raw, cloud_test_raw)
    sig_train, sig_test = standardize(sig_train_raw, sig_test_raw)
    spatial_train, spatial_test = standardize(spatial_train_raw, spatial_test_raw)
    scene_train = np.concatenate([cloud_train, 0.74 * sig_train], axis=1).astype(np.float32)
    scene_test = np.concatenate([cloud_test, 0.74 * sig_test], axis=1).astype(np.float32)
    geom_train = np.concatenate([spatial_train, 0.90 * sig_train], axis=1).astype(np.float32)
    geom_test = np.concatenate([spatial_test, 0.90 * sig_test], axis=1).astype(np.float32)

    print("build global multiscale template targets", flush=True)
    global_targets: list[np.ndarray] = []
    global_probs: list[np.ndarray] = []
    global_reports: dict[str, object] = {}
    for j, k in enumerate(parse_ints(args.cluster_list)):
        tgt, prob, diag = make_template_target(
            truth,
            train_current,
            current,
            scene_train,
            scene_test,
            geom_train,
            geom_test,
            int(k),
            int(args.svd_components),
            int(args.seed) + j * 31,
            args,
            train_mask=None,
        )
        global_targets.append(tgt)
        global_probs.append(prob)
        global_reports[f"k{k}"] = diag

    print("build regime-specific template target", flush=True)
    regime_target, regime_prob_top, regime_report = regime_specific_target(
        truth,
        train_current,
        current,
        scene_train,
        scene_test,
        geom_train,
        geom_test,
        regimes_train,
        regimes_test,
        args,
    )

    # Blend global granularities.  Fine templates get more weight in heavy
    # areas; coarse templates stabilize medium/light.
    global_coeff = np.zeros((len(current), len(global_targets)), dtype=np.float32)
    for i, reg in enumerate(regimes_test):
        if reg == "heavy":
            vals = np.array([0.22, 0.34, 0.44], dtype=np.float32)
        elif reg == "medium":
            vals = np.array([0.34, 0.38, 0.28], dtype=np.float32)
        elif reg == "light":
            vals = np.array([0.45, 0.35, 0.20], dtype=np.float32)
        else:
            vals = np.array([0.55, 0.30, 0.15], dtype=np.float32)
        global_coeff[i, : len(global_targets)] = vals[: len(global_targets)] / np.maximum(vals[: len(global_targets)].sum(), TINY)
    global_target = log_blend(global_targets, global_coeff, float(args.floor), int(args.batch))

    print("test-test smoothing of template target", flush=True)
    test_feat = np.concatenate([scene_test, 0.35 * geom_test], axis=1).astype(np.float32)
    idx_test, dist_test = pairwise_knn(test_feat, test_feat, int(args.k_test) + 1, exclude_self=True)
    idx_test = idx_test[:, : int(args.k_test)]
    dist_test = dist_test[:, : int(args.k_test)]
    w_test = weights_from_dist(dist_test, float(args.test_power), float(args.test_floor))
    smooth_target = log_average(global_target, idx_test, w_test, float(args.floor), int(args.batch))

    agree_global_regime = row_cos(global_target, regime_target).astype(np.float32)
    agree_global_current = row_cos(global_target, current).astype(np.float32)
    agree_regime_current = row_cos(regime_target, current).astype(np.float32)
    agree_smooth_current = row_cos(smooth_target, current).astype(np.float32)
    novelty = np.clip((1.0 - np.minimum(agree_global_current, agree_regime_current)) / 0.16, 0.0, 1.0).astype(np.float32)
    agreement = np.minimum.reduce([agree_global_regime, agree_global_current, agree_regime_current]).astype(np.float32)
    global_top = np.mean([topk_prob(p, 8)[:, 0] for p in global_probs], axis=0).astype(np.float32)
    regime_top = regime_prob_top[:, 0].astype(np.float32)
    conf = (
        0.30 * np.clip((agreement - 0.78) / 0.16, 0.0, 1.0)
        + 0.20 * np.clip((agree_smooth_current - 0.84) / 0.12, 0.0, 1.0)
        + 0.22 * np.clip((global_top - 0.16) / 0.28, 0.0, 1.0)
        + 0.18 * np.clip((regime_top - 0.14) / 0.28, 0.0, 1.0)
        + 0.10 * novelty
    ).astype(np.float32)

    reg_strength = {"clear": 0.010, "light": 0.052, "medium": 0.820, "heavy": 1.080}
    strength = np.array([reg_strength[str(r)] for r in regimes_test], dtype=np.float32)
    strength *= (0.50 + 0.50 * conf).astype(np.float32)
    strength *= (0.58 + 0.42 * np.clip((agreement - 0.70) / 0.24, 0.0, 1.0)).astype(np.float32)
    strength = np.clip(strength, 0.0, 1.12).astype(np.float32)

    coeff = np.zeros((len(current), 7), dtype=np.float32)
    coeff[:, 0] = 1.0 - strength
    coeff[:, 1] = 0.430 * strength
    coeff[:, 2] = 0.355 * strength
    coeff[:, 3] = 0.120 * strength
    coeff[:, 4] = 0.055 * strength
    coeff[:, 5] = 0.028 * strength
    coeff[:, 6] = 0.012 * strength
    target = log_blend(
        [current, global_target, regime_target, smooth_target, v115_anchor, v112_anchor, v111_anchor],
        coeff,
        float(args.floor),
        int(args.batch),
    )

    # Projection confidence should reflect the sharpest stable template
    # decision.  Concatenating all branches dilutes max weight and makes the
    # actual projection too conservative even when the target is high-confidence.
    # Use the finest global template distribution for the projection gate, while
    # the target itself still uses global/regime/smooth ensemble information.
    combined_weights = topk_prob(global_probs[-1], 12)

    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.target_out, target)
    np.save(args.weights_out, combined_weights.astype(np.float32))

    proj = projection(target, combined_weights.astype(np.float32), regimes_test, pos_test, args)
    payload = {
        "method": "v122_multiscale_template_apex",
        "base": args.base,
        "operation": "v121 base; multiscale train-truth support-template ensemble + regime-specific templates + test-test smoothing; phase/scalar/total power preserved",
        "args": vars(args),
        "global_templates": global_reports,
        "regime_templates": regime_report,
        "test_template": {
            "global_vs_current": summarize(agree_global_current),
            "regime_vs_current": summarize(agree_regime_current),
            "global_regime_agreement": summarize(agree_global_regime),
            "smooth_vs_current": summarize(agree_smooth_current),
            "global_top1_mean": summarize(global_top),
            "regime_top1": summarize(regime_top),
            "confidence": summarize(conf),
            "projection_strength": summarize(strength),
        },
        "target_cos": {
            "target_vs_current": summarize(row_cos(target, current)),
            "target_vs_global": summarize(row_cos(target, global_target)),
            "target_vs_regime": summarize(row_cos(target, regime_target)),
            "target_vs_smooth": summarize(row_cos(target, smooth_target)),
            "target_vs_v115": summarize(row_cos(target, v115_anchor)),
            "target_vs_v112": summarize(row_cos(target, v112_anchor)),
            "target_vs_v111": summarize(row_cos(target, v111_anchor)),
        },
        "projection": proj,
        "outputs": {"submission_npy": args.output, "target": args.target_out, "weights": args.weights_out},
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    def write_zip(path: str) -> None:
        files = [
            (args.output, "Round1_Test_Channel.npy"),
            (__file__, "generate_v122_multiscale_template_apex.py"),
            (ASSETS + "/generate_v121_support_template_manifold.py", "generate_v121_support_template_manifold.py"),
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
