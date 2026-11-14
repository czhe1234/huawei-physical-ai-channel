#!/usr/bin/env python3
"""Reconstruct a test-like train v129 target and re-audit the v140 operator."""

from __future__ import annotations

import json
import math

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from analyze_v140_physical_mechanism import (
    circular_center,
    distribution_entropy,
    evaluate_operator,
    gain_report,
    marginal_h,
    marginal_oracles,
    normalize_support,
    normalize_truth,
    row_cos,
    shift_counts,
    summarize,
    transform_batch,
)
from generate_v112_kernel_moe_path_support import (
    entropy01,
    log_blend,
    pairwise_knn,
    softmax_np,
    standardize,
    weights_from_dist,
)
from generate_v115_pointcloud_scene_support import official_scene_embedding, spatial_embedding
from generate_v121_support_template_manifold import support_signature
from generate_v128_badpoint_support_residual_templates import (
    apply_ratio_support,
    build_bad_mask,
    probability_from_bad_neighbors,
)
from generate_v140_balanced_hminus2_overdrive import build_risk, support_features
from generate_v87_easy_field_projection import ad_transform, beta_subcarriers
from powerfield_common import BS_POS, TINY, regime_of
from project_paths import PATHS


REPORT_DIR = PATHS.version_report_dir(140)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v140_source_alignment_analysis.json"
TRAIN_V128 = PATHS.shared_features / "train_v128_testlike_target.npy"
TRAIN_V129 = PATHS.shared_features / "train_v129_testlike_target.npy"
TRAIN_V129_PROB = PATHS.shared_features / "train_v129_testlike_residual_probs.npy"

RANDOM_STATE = 14097


def testlike_features(
    positions: np.ndarray,
    blocked: np.ndarray,
    bank_support: np.ndarray,
    query_support: np.ndarray,
    template_probability: np.ndarray,
    scene: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray]:
    walls = scene["wall_patches"].astype(np.float32)
    cloud_raw = official_scene_embedding(
        positions,
        blocked,
        scene["train_features"].astype(np.float32),
        scene["train_reflection_score"].astype(np.float32),
        scene["train_reflection_path_length"].astype(np.float32),
        scene["train_reflection_excess_length"].astype(np.float32),
        scene["train_reflection_point"].astype(np.float32),
        scene["train_reflection_patch_id"].astype(np.int64),
        walls,
    )
    spatial_raw = spatial_embedding(positions, blocked)
    bank_signature = support_signature(bank_support, 1e-12)
    query_signature = support_signature(query_support, 1e-12)
    cloud_bank, cloud_query = standardize(cloud_raw, cloud_raw)
    spatial_bank, spatial_query = standardize(spatial_raw, spatial_raw)
    signature_bank, signature_query = standardize(bank_signature, query_signature)
    probability_bank, probability_query = standardize(template_probability, template_probability)
    bank = np.concatenate(
        [0.82 * cloud_bank, 0.50 * spatial_bank, 0.72 * signature_bank, 0.42 * probability_bank], axis=1
    ).astype(np.float32)
    query = np.concatenate(
        [0.82 * cloud_query, 0.50 * spatial_query, 0.72 * signature_query, 0.42 * probability_query], axis=1
    ).astype(np.float32)
    return bank, query


def residual_prototypes(
    train_base: np.ndarray,
    truth: np.ndarray,
    bad_mask: np.ndarray,
    labels: np.ndarray,
    ratio_clip: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    bad_base = train_base[bad_mask]
    bad_truth = truth[bad_mask]
    if len(labels) != len(bad_base):
        raise ValueError(f"label count {len(labels)} does not match bad count {len(bad_base)}")
    cluster_count = int(labels.max()) + 1
    log_ratio = np.clip(
        np.log(np.maximum(bad_truth, 1e-12)) - np.log(np.maximum(bad_base, 1e-12)),
        math.log(ratio_clip[0]),
        math.log(ratio_clip[1]),
    ).astype(np.float32)
    proto_ratio = np.empty((cluster_count,) + train_base.shape[1:], dtype=np.float32)
    proto_truth = np.empty_like(proto_ratio)
    global_ratio = log_ratio.mean(axis=0)
    global_truth = np.log(np.maximum(bad_truth, 1e-12)).mean(axis=0)
    for cluster in range(cluster_count):
        selected = labels == cluster
        ratio_value = log_ratio[selected].mean(axis=0) if np.any(selected) else global_ratio
        truth_value = (
            np.log(np.maximum(bad_truth[selected], 1e-12)).mean(axis=0)
            if np.any(selected)
            else global_truth
        )
        proto_ratio[cluster] = np.exp(ratio_value).astype(np.float32)
        value = np.exp(truth_value).astype(np.float32)
        proto_truth[cluster] = value / np.maximum(value.sum(), TINY)
    return proto_ratio, proto_truth


def mixture_ratio(proto_ratio: np.ndarray, probability: np.ndarray) -> np.ndarray:
    output = np.empty((len(probability),) + proto_ratio.shape[1:], dtype=np.float32)
    log_ratio = np.log(np.maximum(proto_ratio, 1e-12)).astype(np.float32)
    for start in range(0, len(probability), 32):
        end = min(start + 32, len(probability))
        output[start:end] = np.exp(
            np.tensordot(probability[start:end], log_ratio, axes=(1, 0))
        ).astype(np.float32)
    return output


def reconstruct_stage(
    name: str,
    current: np.ndarray,
    train_base: np.ndarray,
    truth: np.ndarray,
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
    template_probability: np.ndarray,
    scene: np.lib.npyio.NpzFile,
    params: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    bad_mask, train_cos = build_bad_mask(
        train_base,
        truth,
        regimes,
        float(params["bad_quantile"]),
        float(params["min_bad_cos"]),
    )
    labels = np.load(params["labels"]).astype(np.int64)
    proto_ratio, proto_truth = residual_prototypes(
        train_base, truth, bad_mask, labels, params["ratio_clip"]
    )
    bank_features, query_features = testlike_features(
        positions,
        blocked,
        train_base,
        current,
        template_probability,
        scene,
    )
    bad_indices = np.flatnonzero(bad_mask)
    bad_features = bank_features[bad_mask]
    k = int(params["k_bad"])
    idx_all, dist_all = pairwise_knn(query_features, bad_features, k + 1, exclude_self=False)
    idx = np.empty((len(current), k), dtype=np.int64)
    distance = np.empty((len(current), k), dtype=np.float32)
    for row in range(len(current)):
        keep = bad_indices[idx_all[row]] != row
        idx[row] = idx_all[row][keep][:k]
        distance[row] = dist_all[row][keep][:k]
    weights = weights_from_dist(distance, float(params["bad_power"]), float(params["bad_floor"]))
    probability_bad = probability_from_bad_neighbors(idx, weights, labels, len(proto_ratio))
    proto_flat = proto_truth.reshape(len(proto_truth), -1)
    proto_norm = proto_flat / np.maximum(np.linalg.norm(proto_flat, axis=1, keepdims=True), TINY)
    current_flat = current.reshape(len(current), -1)
    current_norm = current_flat / np.maximum(np.linalg.norm(current_flat, axis=1, keepdims=True), TINY)
    probability_proto = softmax_np(current_norm @ proto_norm.T, temp=float(params["proto_temp"]))
    bad_weight = float(params["bad_weight"])
    proto_weight = float(params["proto_weight"])
    probability = (
        bad_weight * probability_bad + proto_weight * probability_proto
    ) / (bad_weight + proto_weight)
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), TINY)

    ratio = mixture_ratio(proto_ratio, probability)
    residual_target = apply_ratio_support(current, ratio, 1e-12)
    maximum = probability.max(axis=1).astype(np.float32)
    entropy = entropy01(probability).astype(np.float32)
    average_badness = (weights * (1.0 - train_cos[bad_mask][idx])).sum(axis=1).astype(np.float32)
    residual_cos = row_cos(residual_target, current)
    strength = np.clip((average_badness - 0.22) / 0.36, 0.0, 1.0)
    strength *= 0.42 + 0.58 * np.clip((maximum - 0.12) / 0.28, 0.0, 1.0)
    strength *= 0.50 + 0.50 * np.clip((0.94 - entropy) / 0.45, 0.0, 1.0)
    strength *= 0.78 + 0.22 * np.clip((1.0 - residual_cos) / 0.035, 0.0, 1.0)
    regime_scale = np.asarray(
        [{"clear": 0.02, "light": 0.18, "medium": 0.86, "heavy": 1.08}[str(value)] for value in regimes],
        dtype=np.float32,
    )
    strength = np.clip(strength * regime_scale, 0.0, 1.0).astype(np.float32)
    coefficients = np.column_stack(
        [1.0 - strength, 0.82 * strength, 0.10 * strength, 0.055 * strength, 0.025 * strength]
    ).astype(np.float32)
    target = log_blend(
        [current, residual_target, current, current, current], coefficients, 1e-12, 32
    )
    return target.astype(np.float32), probability.astype(np.float32), {
        "name": name,
        "bad_count": int(bad_mask.sum()),
        "clusters": int(len(proto_ratio)),
        "probability_top1": summarize(maximum),
        "probability_entropy": summarize(entropy),
        "nearest_bad_distance": summarize(np.sqrt(np.maximum(distance[:, 0], 0.0))),
        "residual_strength": summarize(strength),
        "target_vs_current_cos": summarize(row_cos(target, current)),
        "target_vs_truth_cos": summarize(row_cos(target, truth)),
    }


def support_summary(
    values: np.ndarray, regimes: np.ndarray | None = None
) -> dict[str, object]:
    h = marginal_h(values)
    _, concentration = circular_center(h)
    flat = values.reshape(len(values), -1)
    h_entropy = distribution_entropy(h)
    joint_top1 = flat.max(axis=1)
    joint_entropy = (
        -(flat * np.log(np.maximum(flat, 1e-30))).sum(axis=1)
        / math.log(flat.shape[1])
    )
    report: dict[str, object] = {
        "h_entropy": summarize(h_entropy),
        "h_concentration": summarize(concentration),
        "joint_top1": summarize(joint_top1),
        "joint_entropy": summarize(joint_entropy),
    }
    if regimes is not None:
        report["by_regime"] = {
            name: {
                "count": int(np.sum(regimes == name)),
                "h_entropy": summarize(h_entropy[regimes == name]),
                "h_concentration": summarize(concentration[regimes == name]),
                "joint_top1": summarize(joint_top1[regimes == name]),
                "joint_entropy": summarize(joint_entropy[regimes == name]),
            }
            for name in ("clear", "light", "medium", "heavy")
        }
    return report


def operator_shape_effect(
    source: np.ndarray,
    risk: np.ndarray,
) -> dict[str, object]:
    active = risk > 0.64
    w1 = np.where(active, 0.165 * risk, 0.0).astype(np.float32)
    w2 = np.where(risk > 0.90, 0.060 * risk, 0.0).astype(np.float32)
    changes = {"log": [], "linear": []}
    for start in range(0, len(source), 64):
        end = min(start + 64, len(source))
        base = source[start:end]
        base_flat = base.reshape(len(base), -1).astype(np.float64)
        base_h = marginal_h(base).astype(np.float64)
        base_joint_entropy = -(base_flat * np.log(np.maximum(base_flat, 1e-30))).sum(axis=1)
        base_h_entropy = -(base_h * np.log(np.maximum(base_h, 1e-30))).sum(axis=1)
        base_values = np.column_stack(
            [
                base_joint_entropy,
                base_h_entropy,
                base_joint_entropy - base_h_entropy,
                base_flat.max(axis=1),
            ]
        )
        for mode in changes:
            output = transform_batch(base, w1[start:end], w2[start:end], mode)
            flat = output.reshape(len(output), -1).astype(np.float64)
            h = marginal_h(output).astype(np.float64)
            joint_entropy = -(flat * np.log(np.maximum(flat, 1e-30))).sum(axis=1)
            h_entropy = -(h * np.log(np.maximum(h, 1e-30))).sum(axis=1)
            output_values = np.column_stack(
                [joint_entropy, h_entropy, joint_entropy - h_entropy, flat.max(axis=1)]
            )
            changes[mode].append(output_values - base_values)
    columns = ("joint_entropy_nat", "h_entropy_nat", "conditional_vd_given_h_entropy_nat", "joint_top1")
    report = {}
    for mode, blocks in changes.items():
        values = np.concatenate(blocks, axis=0)
        report[mode] = {
            name: summarize(values[active, index]) for index, name in enumerate(columns)
        }
    return report


def domain_auc(train: np.ndarray, test: np.ndarray) -> dict[str, float]:
    x = np.concatenate([support_features(train), support_features(test)], axis=0)
    y = np.concatenate([np.zeros(len(train), dtype=np.int32), np.ones(len(test), dtype=np.int32)])
    prediction = np.empty(len(y), dtype=np.float32)
    for train_index, valid_index in StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    ).split(x, y):
        model = ExtraTreesClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x[train_index], y[train_index])
        prediction[valid_index] = model.predict_proba(x[valid_index])[:, 1]
    return {"auc": float(roc_auc_score(y, prediction))}


def collapsed_support_from_complex(path, positions: np.ndarray) -> np.ndarray:
    values = np.load(path, mmap_mode="r")
    output = np.empty((len(values), 16, 8, 192), dtype=np.float32)
    distance = np.linalg.norm(positions - BS_POS[None, :], axis=1)
    beta = beta_subcarriers(140.3, 0.10)
    for start in range(0, len(values), 20):
        end = min(start + 20, len(values))
        block = np.asarray(values[start:end], dtype=np.complex64)
        derotated = block * np.exp(
            1j * distance[start:end, None, None, None] * beta
        ).astype(np.complex64)
        ad = ad_transform(derotated)
        power = (ad.real * ad.real + ad.imag * ad.imag).astype(np.float32)
        collapsed = power.sum(axis=(1, 4))
        output[start:end] = collapsed / np.maximum(
            collapsed.sum(axis=(1, 2, 3), keepdims=True), TINY
        )
    return output


def support_change(before: np.ndarray, after: np.ndarray, regimes: np.ndarray) -> dict[str, object]:
    before_flat = before.reshape(len(before), -1)
    after_flat = after.reshape(len(after), -1)
    before_h = marginal_h(before)
    after_h = marginal_h(after)
    h_entropy_change = distribution_entropy(after_h) - distribution_entropy(before_h)
    joint_entropy_change = (
        -(after_flat * np.log(np.maximum(after_flat, 1e-30))).sum(axis=1)
        + (before_flat * np.log(np.maximum(before_flat, 1e-30))).sum(axis=1)
    ) / math.log(before_flat.shape[1])
    top1_change = after_flat.max(axis=1) - before_flat.max(axis=1)
    return {
        "cosine": summarize(row_cos(before, after)),
        "h_entropy_change": summarize(h_entropy_change),
        "joint_entropy_change": summarize(joint_entropy_change),
        "joint_top1_change": summarize(top1_change),
        "by_regime": {
            name: {
                "h_entropy_change": summarize(h_entropy_change[regimes == name]),
                "joint_entropy_change": summarize(joint_entropy_change[regimes == name]),
                "joint_top1_change": summarize(top1_change[regimes == name]),
            }
            for name in ("clear", "light", "medium", "heavy")
        },
    }


def main() -> None:
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = height["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    scene = np.load(PATHS.shared_features / "official_pointcloud_scene_cache.npz")
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    geodesic_names = [str(value) for value in geodesic["feature_names"]]
    template_probability = np.load(PATHS.shared_features / "train_v125eq_crossfit_template_probs.npy").astype(np.float32)
    train_base = normalize_support(np.load(PATHS.cache_dir(128) / "train_v125eq_collapsed_support.npy"))
    v124like = normalize_support(np.load(PATHS.shared_features / "train_v125eq_crossfit_support_target.npy"))
    raw_truth = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(np.float32)
    truth_power = raw_truth.sum(axis=(1, 2, 3))
    valid = truth_power > 0
    truth = normalize_truth(raw_truth)

    v128_params = {
        "bad_quantile": 0.58,
        "min_bad_cos": 0.965,
        "labels": PATHS.cache_dir(128) / "v128_badpoint_residual_train_labels.npy",
        "ratio_clip": (0.20, 5.20),
        "k_bad": 34,
        "bad_power": 3.15,
        "bad_floor": 0.42,
        "proto_temp": 0.018,
        "bad_weight": 0.76,
        "proto_weight": 0.24,
    }
    v129_params = {
        "bad_quantile": 0.52,
        "min_bad_cos": 0.975,
        "labels": PATHS.cache_dir(129) / "v129_apex_badpoint_residual_train_labels.npy",
        "ratio_clip": (0.16, 6.80),
        "k_bad": 46,
        "bad_power": 3.65,
        "bad_floor": 0.34,
        "proto_temp": 0.014,
        "bad_weight": 0.84,
        "proto_weight": 0.16,
    }
    train_v128, probability_v128, report_v128 = reconstruct_stage(
        "v128", v124like, train_base, truth, positions, blocked, regimes,
        template_probability, scene, v128_params
    )
    np.save(TRAIN_V128, train_v128)
    train_v129, probability_v129, report_v129 = reconstruct_stage(
        "v129", train_v128, train_base, truth, positions, blocked, regimes,
        template_probability, scene, v129_params
    )
    np.save(TRAIN_V129, train_v129)
    np.save(TRAIN_V129_PROB, probability_v129)

    risk = build_risk(regimes, probability_v129, geodesic["train_features"], geodesic_names)
    active = risk > 0.64
    extreme = risk > 0.90
    w1 = np.where(active, 0.165 * risk, 0.0).astype(np.float32)
    w2 = np.where(extreme, 0.060 * risk, 0.0).astype(np.float32)
    v129_score = row_cos(train_v129, truth)
    log_value = evaluate_operator(train_v129, truth, w1, w2, "log")
    linear_value = evaluate_operator(train_v129, truth, w1, w2, "linear")
    h_oracle = marginal_oracles(marginal_h(train_v129), marginal_h(truth))

    test_v129 = normalize_support(
        np.load(PATHS.cache_dir(129) / "v129_apex_badpoint_support_residual_target.npy")
    )
    test_probability = np.load(
        PATHS.cache_dir(129) / "v129_apex_badpoint_support_residual_probs.npy"
    ).astype(np.float32)
    test_positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    test_blocked = height["test_blocked"].astype(np.float32)
    test_regimes = regime_of(test_blocked)
    test_risk = build_risk(
        test_regimes, test_probability, geodesic["test_features"], geodesic_names
    )
    test_w1 = np.where(test_risk > 0.64, 0.165 * test_risk, 0.0).astype(np.float32)
    test_w2 = np.where(test_risk > 0.90, 0.060 * test_risk, 0.0).astype(np.float32)
    dummy_truth = test_v129
    test_log = evaluate_operator(test_v129, dummy_truth, test_w1, test_w2, "log")
    train_h = marginal_h(train_v129)
    test_h = marginal_h(test_v129)
    train_center, _ = circular_center(train_h)
    test_center, _ = circular_center(test_h)
    train_log_center, _ = circular_center(log_value["h"])
    test_log_center, _ = circular_center(test_log["h"])

    v129_output_support = collapsed_support_from_complex(
        PATHS.cache_dir(129) / "v129_Round1_Test_Channel.npy", test_positions
    )
    v140_output_support = collapsed_support_from_complex(
        PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", test_positions
    )
    v142_output_support = collapsed_support_from_complex(
        PATHS.cache_dir(142) / "v142_Round1_Test_Channel.npy", test_positions
    )
    v140_target = normalize_support(
        np.load(PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_target.npy")
    )

    cached_matched = np.load(PATHS.shared_features / "train_v129_crossfit_residual_probs.npy").astype(np.float32)
    probability_cos = row_cos(probability_v129, cached_matched)
    report = {
        "method": "v140_source_alignment_analysis",
        "status": "analysis_only_no_candidate",
        "question": (
            "Does v140 correct a propagation law, or a systematic bias introduced by the recursive "
            "v128/v129 residual-template target that was absent from the previous train analogue?"
        ),
        "reconstruction_limit": (
            "Queries use test-like current-support features and exclude themselves from bad-neighbor "
            "selection. Cluster templates still use aggregate train truth, and unavailable v121/v123 "
            "train anchors are approximated by the v124-equivalent cross-fit support target."
        ),
        "stages": {"v128": report_v128, "v129": report_v129},
        "source_scores": {
            "v125_collapsed": summarize(row_cos(train_base, truth)[valid]),
            "v124like_target": summarize(row_cos(v124like, truth)[valid]),
            "v128_testlike_target": summarize(row_cos(train_v128, truth)[valid]),
            "v129_testlike_target": summarize(v129_score[valid]),
        },
        "support_distributions": {
            "train_truth": support_summary(truth, regimes),
            "train_v125_collapsed": support_summary(train_base, regimes),
            "train_v124like": support_summary(v124like, regimes),
            "train_v128_testlike": support_summary(train_v128, regimes),
            "train_v129_testlike": support_summary(train_v129, regimes),
            "test_v129": support_summary(test_v129, test_regimes),
            "train_test_support_domain": domain_auc(train_v129, test_v129),
        },
        "probability_alignment": {
            "testlike_vs_previous_matched_cos": summarize(probability_cos),
            "testlike_top1": summarize(probability_v129.max(axis=1)),
            "previous_matched_top1": summarize(cached_matched.max(axis=1)),
            "test_top1": summarize(test_probability.max(axis=1)),
        },
        "v140_operator_on_testlike_train_source": {
            "risk": summarize(risk),
            "active_fraction": float(np.mean(active)),
            "extreme_fraction": float(np.mean(extreme)),
            "log_gain": gain_report(log_value["score"] - v129_score, valid, regimes, risk),
            "linear_gain": gain_report(linear_value["score"] - v129_score, valid, regimes, risk),
            "h_oracle": {
                "shift_gain": summarize((h_oracle["best_shift_score"] - h_oracle["base_score"])[valid]),
                "best_shift_counts": shift_counts(h_oracle["best_shift"], valid),
                "active_best_shift_negative": float(np.mean(h_oracle["best_shift"][valid & active] < 0)),
                "active_best_shift_zero": float(np.mean(h_oracle["best_shift"][valid & active] == 0)),
                "active_best_shift_positive": float(np.mean(h_oracle["best_shift"][valid & active] > 0)),
            },
        },
        "operator_shape_effect": {
            "interpretation": (
                "Positive entropy deltas and negative top1 deltas mean the operator is directional "
                "multiplicative diffusion, not sharpening."
            ),
            "full_shape_train": operator_shape_effect(train_v129, risk),
            "full_shape_test": operator_shape_effect(test_v129, test_risk),
            "train_active_h_entropy_change": summarize(
                (distribution_entropy(log_value["h"]) - distribution_entropy(train_h))[active]
            ),
            "test_active_h_entropy_change": summarize(
                (distribution_entropy(test_log["h"]) - distribution_entropy(test_h))[test_risk > 0.64]
            ),
            "train_active_center_shift": summarize(
                ((train_log_center - train_center + 8.0) % 16.0 - 8.0)[active]
            ),
            "test_active_center_shift": summarize(
                ((test_log_center - test_center + 8.0) % 16.0 - 8.0)[test_risk > 0.64]
            ),
        },
        "actual_complex_output_support": {
            "v129_output": support_summary(v129_output_support, test_regimes),
            "v140_output": support_summary(v140_output_support, test_regimes),
            "v142_output": support_summary(v142_output_support, test_regimes),
            "v129_target_vs_output_cos": summarize(row_cos(test_v129, v129_output_support)),
            "v140_target_vs_output_cos": summarize(row_cos(v140_target, v140_output_support)),
            "v140_output_change_from_v129": support_change(
                v129_output_support, v140_output_support, test_regimes
            ),
            "v142_output_change_from_v129": support_change(
                v129_output_support, v142_output_support, test_regimes
            ),
            "v140_vs_v142_interpretation": (
                "v140 raises H entropy more while retaining higher cosine to v129 than v142. "
                "Geometric log overlap therefore adds adjacent uncertainty with less main-support "
                "displacement than linear mass transport."
            ),
            "train_v125eq_vs_test_v129_output_domain": domain_auc(
                train_base, v129_output_support
            ),
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "train_v128_target": PATHS.relative(TRAIN_V128),
            "train_v129_target": PATHS.relative(TRAIN_V129),
            "train_v129_probability": PATHS.relative(TRAIN_V129_PROB),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {REPORT}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
