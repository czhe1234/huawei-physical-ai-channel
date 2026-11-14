#!/usr/bin/env python3
"""Generate v121: support-template manifold projection.

This version is a deliberate break from the failed v120 risky-template line.
Instead of assigning test points to explicit inverse-RT path/component templates,
it mines propagation-mode templates directly from the training truth support:

    train true collapsed AD support -> cluster/prototype manifold

Then each test point gets a soft template distribution from two independent
signals:

1. official point-cloud/geometry similarity to training points;
2. similarity between the current v115 support and the learned prototypes.

The final target only moves v115 toward the prototype manifold when these two
signals agree.  Phase, complex scalar structure, and total power are still
preserved; only normalized collapsed joint AD support is nudged.
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
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


REGIMES = ("clear", "light", "medium", "heavy")


def support_signature(supp: np.ndarray, floor: float) -> np.ndarray:
    n = supp.shape[0]
    flat = supp.reshape(n, -1).astype(np.float32)
    beam = supp.sum(axis=-1).reshape(n, -1).astype(np.float32)
    delay = supp.sum(axis=(1, 2)).astype(np.float32)
    top = np.sort(flat, axis=1)[:, ::-1]
    top1 = top[:, 0]
    top4 = top[:, :4].sum(axis=1)
    top16 = top[:, :16].sum(axis=1)
    top64 = top[:, :64].sum(axis=1)
    ent = -(flat * np.log(np.maximum(flat, floor))).sum(axis=1) / math.log(flat.shape[1])
    beam_ent = -(beam * np.log(np.maximum(beam, floor))).sum(axis=1) / math.log(beam.shape[1])
    delay_ent = -(delay * np.log(np.maximum(delay, floor))).sum(axis=1) / math.log(delay.shape[1])

    h = np.arange(MH, dtype=np.float32)
    v = np.arange(MV, dtype=np.float32)
    t = np.arange(S_SC, dtype=np.float32)
    bh = supp.sum(axis=(2, 3))
    bv = supp.sum(axis=(1, 3))
    bt = delay
    h_mean = (bh * h[None, :]).sum(axis=1) / np.maximum(bh.sum(axis=1), TINY)
    v_mean = (bv * v[None, :]).sum(axis=1) / np.maximum(bv.sum(axis=1), TINY)
    t_mean = (bt * t[None, :]).sum(axis=1) / np.maximum(bt.sum(axis=1), TINY)
    h_var = (bh * (h[None, :] - h_mean[:, None]) ** 2).sum(axis=1) / np.maximum(bh.sum(axis=1), TINY)
    v_var = (bv * (v[None, :] - v_mean[:, None]) ** 2).sum(axis=1) / np.maximum(bv.sum(axis=1), TINY)
    t_var = (bt * (t[None, :] - t_mean[:, None]) ** 2).sum(axis=1) / np.maximum(bt.sum(axis=1), TINY)

    stats = np.column_stack(
        [
            top1,
            top4,
            top16,
            top64,
            ent,
            beam_ent,
            delay_ent,
            h_mean / MH,
            v_mean / MV,
            t_mean / S_SC,
            np.sqrt(np.maximum(h_var, 0.0)) / MH,
            np.sqrt(np.maximum(v_var, 0.0)) / MV,
            np.sqrt(np.maximum(t_var, 0.0)) / S_SC,
            beam.max(axis=1),
            delay.max(axis=1),
        ]
    ).astype(np.float32)
    return np.nan_to_num(
        np.concatenate([np.log(np.maximum(beam, floor)), np.log(np.maximum(delay, floor)), stats], axis=1),
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    ).astype(np.float32)


def fit_support_svd(x: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Low-rank support embedding using deterministic randomized SVD if available."""
    try:
        from sklearn.decomposition import TruncatedSVD

        svd = TruncatedSVD(n_components=n_components, random_state=seed, n_iter=7)
        emb = svd.fit_transform(x).astype(np.float32)
        return emb, svd.components_.astype(np.float32), np.asarray(svd.explained_variance_ratio_, dtype=np.float32)
    except Exception:
        # Fallback: dense SVD on centered data.  This is slower but dependency-free.
        xc = x - x.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(xc, full_matrices=False)
        k = min(n_components, vt.shape[0])
        emb = (u[:, :k] * s[:k]).astype(np.float32)
        return emb, vt[:k].astype(np.float32), np.zeros((k,), dtype=np.float32)


def cluster_embeddings(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    try:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=512,
            n_init=8,
            max_iter=500,
            reassignment_ratio=0.02,
        )
        return km.fit_predict(x).astype(np.int64)
    except Exception:
        rng = np.random.default_rng(seed)
        centers = x[rng.choice(len(x), size=k, replace=False)].astype(np.float32)
        labels = np.zeros(len(x), dtype=np.int64)
        for _ in range(60):
            d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(d, axis=1).astype(np.int64)
            for j in range(k):
                m = labels == j
                if np.any(m):
                    centers[j] = x[m].mean(axis=0)
        return labels


def build_prototypes(truth: np.ndarray, labels: np.ndarray, k: int, floor: float) -> np.ndarray:
    proto = np.empty((k, MH, MV, S_SC), dtype=np.float32)
    global_log = np.log(np.maximum(truth, floor)).mean(axis=0)
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            p = np.exp(global_log).astype(np.float32)
        else:
            p = np.exp(np.log(np.maximum(truth[idx], floor)).mean(axis=0)).astype(np.float32)
        p /= np.maximum(p.sum(), TINY)
        proto[j] = p
    return proto


def template_blend(prototypes: np.ndarray, prob: np.ndarray, floor: float, batch: int) -> np.ndarray:
    n = prob.shape[0]
    out = np.empty((n, MH, MV, S_SC), dtype=np.float32)
    logp = np.log(np.maximum(prototypes, floor)).astype(np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        mix = np.tensordot(prob[s:e].astype(np.float32), logp, axes=(1, 0)).astype(np.float32)
        pred = np.exp(mix).astype(np.float32)
        pred /= np.maximum(pred.sum(axis=(1, 2, 3), keepdims=True), TINY)
        out[s:e] = pred
    return out


def topk_prob(prob: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, prob.shape[1])
    ids = np.argpartition(prob, -kk, axis=1)[:, -kk:]
    vals = np.take_along_axis(prob, ids, axis=1)
    order = np.argsort(vals, axis=1)[:, ::-1]
    vals = np.take_along_axis(vals, order, axis=1)
    vals /= np.maximum(vals.sum(axis=1, keepdims=True), TINY)
    return vals.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--scene-cache", default=ASSETS + "/official_pointcloud_scene_cache.npz")
    ap.add_argument("--base", default=ASSETS + "/cache_v115/v115_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=ASSETS + "/cache_v121/v121_Round1_Test_Channel.npy")
    ap.add_argument("--truth-support", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--train-current-support", default=ASSETS + "/train_v98like_loo_joint_target.npy")
    ap.add_argument("--current-target", default=ASSETS + "/cache_v115/v115_pointcloud_scene_support_target.npy")
    ap.add_argument("--v112-anchor", default=ASSETS + "/cache_v112/v112_kernel_moe_path_support_target.npy")
    ap.add_argument("--v111-anchor", default=ASSETS + "/cache_v111/v111_manual_kernel_path_support_target.npy")
    ap.add_argument("--v109-anchor", default=ASSETS + "/cache_v109/v109_accelerated_momentum_path_support_target.npy")
    ap.add_argument("--target-out", default=ASSETS + "/cache_v121/v121_support_template_manifold_target.npy")
    ap.add_argument("--weights-out", default=ASSETS + "/cache_v121/v121_support_template_manifold_template_weights.npy")
    ap.add_argument("--prob-out", default=ASSETS + "/cache_v121/v121_support_template_manifold_probs.npy")
    ap.add_argument("--label-out", default=ASSETS + "/cache_v121/v121_support_template_manifold_train_labels.npy")
    ap.add_argument("--report", default=ASSETS + "/v121_support_template_manifold_generate.json")
    ap.add_argument("--zip-out", default=str(PATHS.round1_versions / "v121_support_template_manifold_candidate_pending.zip"))
    ap.add_argument("--candidate-zip", default=str(PATHS.round1_versions / "Round1_Submission_candidate_v121_support_template_manifold.zip"))
    ap.add_argument("--clusters", type=int, default=96)
    ap.add_argument("--svd-components", type=int, default=72)
    ap.add_argument("--seed", type=int, default=12107)
    ap.add_argument("--k-geom", type=int, default=28)
    ap.add_argument("--k-scene", type=int, default=18)
    ap.add_argument("--geom-power", type=float, default=2.65)
    ap.add_argument("--scene-power", type=float, default=3.10)
    ap.add_argument("--geom-floor", type=float, default=0.62)
    ap.add_argument("--scene-floor", type=float, default=0.48)
    ap.add_argument("--proto-temp", type=float, default=0.020)
    ap.add_argument("--scene-weight", type=float, default=0.52)
    ap.add_argument("--geom-weight", type=float, default=0.26)
    ap.add_argument("--current-weight", type=float, default=0.22)
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project-batch", type=int, default=10)
    ap.add_argument("--eta", default="clear:0.0005,light:0.0045,medium:0.780,heavy:1.060")
    ap.add_argument("--min-gate", type=float, default=0.20)
    ap.add_argument("--max-gate", type=float, default=2.35)
    ap.add_argument("--conf-center", type=float, default=0.18)
    ap.add_argument("--conf-width", type=float, default=0.15)
    ap.add_argument("--entropy-low", type=float, default=0.20)
    ap.add_argument("--entropy-high", type=float, default=0.92)
    ap.add_argument("--gain-clip", default="0.38,3.45")
    ap.add_argument("--top-gain-clip", default="0.52,2.55")
    ap.add_argument("--top-components", type=int, default=2432)
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
    v112_anchor = normalize_support(np.load(args.v112_anchor), float(args.floor))
    v111_anchor = normalize_support(np.load(args.v111_anchor), float(args.floor))
    v109_anchor = normalize_support(np.load(args.v109_anchor), float(args.floor))

    print("cluster training truth support into manifold templates", flush=True)
    support_x = np.log(np.maximum(truth.reshape(len(truth), -1), float(args.floor))).astype(np.float32)
    # Compress log-support before clustering.  Clustering the full 24k-D tensor
    # directly overfits tiny bins; this captures support mode, not noise.
    emb, _, evr = fit_support_svd(support_x, int(args.svd_components), int(args.seed))
    labels = cluster_embeddings(emb, int(args.clusters), int(args.seed))
    prototypes = build_prototypes(truth, labels, int(args.clusters), float(args.floor))

    print("build official point-cloud and support-signature features", flush=True)
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
    scene_train = np.concatenate([cloud_train, 0.65 * sig_train], axis=1).astype(np.float32)
    scene_test = np.concatenate([cloud_test, 0.65 * sig_test], axis=1).astype(np.float32)
    geom_train = np.concatenate([spatial_train, 0.80 * sig_train], axis=1).astype(np.float32)
    geom_test = np.concatenate([spatial_test, 0.80 * sig_test], axis=1).astype(np.float32)

    print("predict template distribution by geometry kNN and current-support similarity", flush=True)
    idx_scene_tr, dist_scene_tr = pairwise_knn(scene_train, scene_train, int(args.k_scene), exclude_self=True)
    idx_geom_tr, dist_geom_tr = pairwise_knn(geom_train, geom_train, int(args.k_geom), exclude_self=True)
    idx_scene_te, dist_scene_te = pairwise_knn(scene_test, scene_train, int(args.k_scene), exclude_self=False)
    idx_geom_te, dist_geom_te = pairwise_knn(geom_test, geom_train, int(args.k_geom), exclude_self=False)
    w_scene_tr = weights_from_dist(dist_scene_tr, float(args.scene_power), float(args.scene_floor))
    w_geom_tr = weights_from_dist(dist_geom_tr, float(args.geom_power), float(args.geom_floor))
    w_scene_te = weights_from_dist(dist_scene_te, float(args.scene_power), float(args.scene_floor))
    w_geom_te = weights_from_dist(dist_geom_te, float(args.geom_power), float(args.geom_floor))

    k = int(args.clusters)
    eye = np.eye(k, dtype=np.float32)
    prob_scene_tr = (w_scene_tr[:, :, None] * eye[labels[idx_scene_tr]]).sum(axis=1)
    prob_geom_tr = (w_geom_tr[:, :, None] * eye[labels[idx_geom_tr]]).sum(axis=1)
    prob_scene_te = (w_scene_te[:, :, None] * eye[labels[idx_scene_te]]).sum(axis=1)
    prob_geom_te = (w_geom_te[:, :, None] * eye[labels[idx_geom_te]]).sum(axis=1)

    proto_cur_cos = row_cos(
        np.repeat(prototypes[None, 0], len(current), axis=0), current
    )  # placeholder for allocation-free shape check
    proto_flat = prototypes.reshape(k, -1).astype(np.float32)
    cur_flat = current.reshape(len(current), -1).astype(np.float32)
    proto_norm = proto_flat / np.maximum(np.linalg.norm(proto_flat, axis=1, keepdims=True), TINY)
    cur_norm = cur_flat / np.maximum(np.linalg.norm(cur_flat, axis=1, keepdims=True), TINY)
    proto_cur_cos = cur_norm @ proto_norm.T
    prob_cur_te = softmax_np(proto_cur_cos, temp=float(args.proto_temp))

    tr_flat = train_current.reshape(len(train_current), -1).astype(np.float32)
    tr_norm = tr_flat / np.maximum(np.linalg.norm(tr_flat, axis=1, keepdims=True), TINY)
    proto_train_cos = tr_norm @ proto_norm.T
    prob_cur_tr = softmax_np(proto_train_cos, temp=float(args.proto_temp))

    sw = float(args.scene_weight)
    gw = float(args.geom_weight)
    cw = float(args.current_weight)
    norm = max(sw + gw + cw, 1e-6)
    prob_tr = (sw * prob_scene_tr + gw * prob_geom_tr + cw * prob_cur_tr) / norm
    prob_te = (sw * prob_scene_te + gw * prob_geom_te + cw * prob_cur_te) / norm
    prob_tr /= np.maximum(prob_tr.sum(axis=1, keepdims=True), TINY)
    prob_te /= np.maximum(prob_te.sum(axis=1, keepdims=True), TINY)

    print("blend template prototypes", flush=True)
    train_template_target = template_blend(prototypes, prob_tr, float(args.floor), int(args.batch))
    template_target = template_blend(prototypes, prob_te, float(args.floor), int(args.batch))

    scene_cur_agree = np.sum(np.sqrt(np.maximum(prob_scene_te, 0.0) * np.maximum(prob_cur_te, 0.0)), axis=1).astype(np.float32)
    geom_scene_agree = np.sum(np.sqrt(np.maximum(prob_scene_te, 0.0) * np.maximum(prob_geom_te, 0.0)), axis=1).astype(np.float32)
    template_cos_current = row_cos(template_target, current).astype(np.float32)
    train_template_cos = row_cos(train_template_target, truth).astype(np.float32)
    train_current_cos = row_cos(train_current, truth).astype(np.float32)
    prob_conf = np.clip((topk_prob(prob_te, 8)[:, 0] - 0.10) / 0.22, 0.0, 1.0).astype(np.float32)

    reg_strength = {"clear": 0.018, "light": 0.055, "medium": 0.680, "heavy": 0.860}
    strength = np.array([reg_strength[str(r)] for r in regimes_test], dtype=np.float32)
    strength *= (0.42 + 0.58 * np.clip((scene_cur_agree - 0.22) / 0.36, 0.0, 1.0)).astype(np.float32)
    strength *= (0.52 + 0.48 * np.clip((geom_scene_agree - 0.18) / 0.34, 0.0, 1.0)).astype(np.float32)
    strength *= (0.46 + 0.54 * np.clip((template_cos_current - 0.84) / 0.11, 0.0, 1.0)).astype(np.float32)
    strength *= (0.56 + 0.44 * prob_conf).astype(np.float32)
    strength = np.clip(strength, 0.0, 0.94).astype(np.float32)

    coeff = np.zeros((len(current), 5), dtype=np.float32)
    coeff[:, 0] = 1.0 - strength
    coeff[:, 1] = 0.88 * strength
    coeff[:, 2] = 0.070 * strength
    coeff[:, 3] = 0.035 * strength
    coeff[:, 4] = 0.015 * strength
    target = log_blend([current, template_target, v112_anchor, v111_anchor, v109_anchor], coeff, float(args.floor), int(args.batch))

    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.target_out, target)
    np.save(args.weights_out, topk_prob(prob_te, 12))
    np.save(args.prob_out, prob_te.astype(np.float32))
    np.save(args.label_out, labels.astype(np.int64))

    proj = projection(target, topk_prob(prob_te, 12), regimes_test, pos_test, args)
    counts = np.bincount(labels, minlength=k)
    payload = {
        "method": "v121_support_template_manifold",
        "base": args.base,
        "operation": "cluster train truth collapsed AD support into propagation-mode prototypes; predict soft test template with official point-cloud geometry + current-support similarity; preserve phase/scalar/total power",
        "args": vars(args),
        "template": {
            "clusters": int(k),
            "cluster_count_min": int(counts.min()),
            "cluster_count_max": int(counts.max()),
            "cluster_count_p10": float(np.quantile(counts, 0.10)),
            "cluster_count_p50": float(np.quantile(counts, 0.50)),
            "cluster_count_p90": float(np.quantile(counts, 0.90)),
            "svd_explained_variance_sum": float(np.sum(evr)),
        },
        "train_diagnostic": {
            "template_target_vs_truth": summarize(train_template_cos),
            "train_current_vs_truth": summarize(train_current_cos),
            "template_minus_current_cos": summarize(train_template_cos - train_current_cos),
        },
        "test_template": {
            "prob_top1": summarize(topk_prob(prob_te, 8)[:, 0]),
            "prob_entropy": summarize(entropy01(prob_te)),
            "scene_cur_agreement": summarize(scene_cur_agree),
            "geom_scene_agreement": summarize(geom_scene_agree),
            "template_vs_current": summarize(template_cos_current),
            "projection_strength": summarize(strength),
        },
        "target_cos": {
            "target_vs_current": summarize(row_cos(target, current)),
            "template_vs_current": summarize(row_cos(template_target, current)),
            "target_vs_template": summarize(row_cos(target, template_target)),
            "target_vs_v112": summarize(row_cos(target, v112_anchor)),
            "target_vs_v111": summarize(row_cos(target, v111_anchor)),
            "target_vs_v109": summarize(row_cos(target, v109_anchor)),
        },
        "projection": proj,
        "outputs": {
            "submission_npy": args.output,
            "target": args.target_out,
            "weights": args.weights_out,
            "prob": args.prob_out,
            "labels": args.label_out,
        },
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    def write_zip(path: str) -> None:
        files = [
            (args.output, "Round1_Test_Channel.npy"),
            (__file__, "generate_v121_support_template_manifold.py"),
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
