#!/usr/bin/env python3
"""Rebuild train-side v129 residual probabilities and redo v143 diagnostics."""

from __future__ import annotations

import json

import numpy as np

from analyze_post_v143_hidden_information import (
    domain_analysis,
    regression_oof,
    safe_corr,
    summarize,
    support_score,
)
from generate_v112_kernel_moe_path_support import (
    normalize_support,
    pairwise_knn,
    softmax_np,
    weights_from_dist,
)
from generate_v128_badpoint_support_residual_templates import (
    build_bad_mask,
    build_features,
    probability_from_bad_neighbors,
)
from generate_v140_balanced_hminus2_overdrive import build_risk, support_features
from generate_v143_transductive_geodesic_risk import (
    apply_risk_shift,
    diffuse_risk,
    graph_average,
    official_graph_embedding,
    robust_standardize,
)
from powerfield_common import TINY, regime_of
from project_paths import PATHS


REPORT_DIR = PATHS.version_report_dir(143)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v143_matched_residual_probability_analysis.json"
TRAIN_PROB_OUT = PATHS.shared_features / "train_v129_crossfit_residual_probs.npy"
RANDOM_STATE = 14391


def entropy01(probabilities: np.ndarray) -> np.ndarray:
    return (
        -(
            probabilities
            * np.log(np.maximum(probabilities, 1e-12))
        ).sum(axis=1)
        / np.log(probabilities.shape[1])
    ).astype(np.float32)


def basic_features(
    positions: np.ndarray, blocked: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    distance = np.linalg.norm(
        positions[:, :2] - np.asarray([50.0, 0.0], dtype=np.float32), axis=1
    )
    azimuth = np.arctan2(positions[:, 1], positions[:, 0] - 50.0)
    values = np.column_stack(
        [
            positions[:, 0],
            positions[:, 1],
            distance,
            np.sin(azimuth),
            np.cos(azimuth),
            blocked,
        ]
    ).astype(np.float32)
    return values, ["x", "y", "distance_bs", "sin_az", "cos_az", "blocked"]


def matched_train_residual_probabilities(
    train_base: np.ndarray,
    test_current: np.ndarray,
    truth: np.ndarray,
    regimes: np.ndarray,
    pos_train: np.ndarray,
    pos_test: np.ndarray,
    blocked_train: np.ndarray,
    blocked_test: np.ndarray,
    train_template_prob: np.ndarray,
    test_template_prob: np.ndarray,
    scene: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, dict[str, object]]:
    bad_mask, train_cos = build_bad_mask(train_base, truth, regimes, 0.52, 0.975)
    bad_indices = np.flatnonzero(bad_mask)
    labels = np.load(
        PATHS.cache_dir(129) / "v129_apex_badpoint_residual_train_labels.npy"
    ).astype(np.int64)
    if len(labels) != len(bad_indices):
        raise ValueError(
            f"v129 bad-mask mismatch: labels={len(labels)} mask={len(bad_indices)}"
        )
    cluster_count = int(labels.max()) + 1

    feat_train, _, feature_report = build_features(
        pos_train,
        pos_test,
        blocked_train,
        blocked_test,
        train_base,
        test_current,
        train_template_prob,
        test_template_prob,
        scene,
        1e-12,
    )
    bad_features = feat_train[bad_mask]
    idx_all, dist_all = pairwise_knn(
        feat_train, bad_features, 47, exclude_self=False
    )
    idx = np.empty((len(train_base), 46), dtype=np.int64)
    dist = np.empty((len(train_base), 46), dtype=np.float32)
    self_removed = 0
    for row in range(len(train_base)):
        keep = bad_indices[idx_all[row]] != row
        available_idx = idx_all[row][keep]
        available_dist = dist_all[row][keep]
        if len(available_idx) < 46:
            raise ValueError(f"too few non-self residual neighbors for train row {row}")
        idx[row] = available_idx[:46]
        dist[row] = available_dist[:46]
        self_removed += int(not np.all(keep))
    weights = weights_from_dist(dist, 3.65, 0.34)
    probability_bad = probability_from_bad_neighbors(
        idx, weights, labels, cluster_count
    )

    bad_truth = truth[bad_mask]
    prototypes = np.empty(
        (cluster_count,) + tuple(truth.shape[1:]), dtype=np.float32
    )
    global_log = np.log(np.maximum(bad_truth, 1e-12)).mean(axis=0)
    for cluster in range(cluster_count):
        selected = labels == cluster
        log_value = (
            np.log(np.maximum(bad_truth[selected], 1e-12)).mean(axis=0)
            if np.any(selected)
            else global_log
        )
        value = np.exp(log_value).astype(np.float32)
        value /= np.maximum(value.sum(), TINY)
        prototypes[cluster] = value

    proto_flat = prototypes.reshape(cluster_count, -1)
    proto_norm = proto_flat / np.maximum(
        np.linalg.norm(proto_flat, axis=1, keepdims=True), TINY
    )
    current_flat = train_base.reshape(len(train_base), -1)
    current_norm = current_flat / np.maximum(
        np.linalg.norm(current_flat, axis=1, keepdims=True), TINY
    )
    probability_proto = softmax_np(current_norm @ proto_norm.T, temp=0.014)
    probabilities = 0.84 * probability_bad + 0.16 * probability_proto
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), TINY)
    np.save(TRAIN_PROB_OUT, probabilities.astype(np.float32))

    return probabilities.astype(np.float32), {
        "bad_count": int(bad_mask.sum()),
        "clusters": cluster_count,
        "self_neighbor_removed_count": self_removed,
        "feature_report": feature_report,
        "train_base_cos": summarize(train_cos),
        "nearest_bad_distance": summarize(np.sqrt(np.maximum(dist[:, 0], 0.0))),
    }


def domain_variant(
    name: str,
    train: np.ndarray,
    test: np.ndarray,
    names: list[str],
) -> dict[str, object]:
    report, _, _ = domain_analysis(
        np.nan_to_num(train, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32),
        np.nan_to_num(test, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32),
        names,
        {name: (0, train.shape[1])},
    )
    return report


def main() -> None:
    pos_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked_train = height["train_blocked"].astype(np.float32)
    blocked_test = height["test_blocked"].astype(np.float32)
    regimes_train = regime_of(blocked_train)
    regimes_test = regime_of(blocked_test)
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    geodesic_names = [str(value) for value in geodesic["feature_names"]]
    scene = np.load(PATHS.shared_features / "official_pointcloud_scene_cache.npz")
    pointcloud_names = [str(value) for value in scene["feature_names"]]

    train_base = normalize_support(
        np.load(PATHS.cache_dir(128) / "train_v125eq_collapsed_support.npy"),
        1e-12,
    )
    test_current = normalize_support(
        np.load(PATHS.cache_dir(128) / "v128_badpoint_support_residual_target.npy"),
        1e-12,
    )
    truth = normalize_support(
        np.load(PATHS.shared_features / "truth_coll_power.npy"), 1e-12
    )
    truth_power = np.load(PATHS.shared_features / "truth_coll_power.npy").sum(
        axis=(1, 2, 3)
    )
    valid = truth_power > 0
    train_v125_prob = np.load(
        PATHS.shared_features / "train_v125eq_crossfit_template_probs.npy"
    ).astype(np.float32)
    test_v124_prob = np.load(
        PATHS.cache_dir(124) / "v124_visibility_misleading_overdrive_probs.npy"
    ).astype(np.float32)
    test_v129_prob = np.load(
        PATHS.cache_dir(129) / "v129_apex_badpoint_support_residual_probs.npy"
    ).astype(np.float32)

    matched_prob, reconstruction = matched_train_residual_probabilities(
        train_base,
        test_current,
        truth,
        regimes_train,
        pos_train,
        pos_test,
        blocked_train,
        blocked_test,
        train_v125_prob,
        test_v124_prob,
        scene,
    )

    raw_mismatched = build_risk(
        regimes_train,
        train_v125_prob,
        geodesic["train_features"],
        geodesic_names,
    )
    raw_matched = build_risk(
        regimes_train,
        matched_prob,
        geodesic["train_features"],
        geodesic_names,
    )
    raw_test = build_risk(
        regimes_test,
        test_v129_prob,
        geodesic["test_features"],
        geodesic_names,
    )
    embedding_train = official_graph_embedding(
        pos_train,
        geodesic["train_features"],
        scene["train_features"],
        matched_prob,
    )
    embedding_test = official_graph_embedding(
        pos_test,
        geodesic["test_features"],
        scene["test_features"],
        test_v129_prob,
    )
    embedding_train, embedding_test = robust_standardize(
        embedding_train, embedding_test
    )
    average_train, disagreement_train, _, _ = graph_average(
        embedding_train, raw_matched, 24
    )
    graph_train = diffuse_risk(raw_matched, average_train, regimes_train)

    mismatched_target, _ = apply_risk_shift(train_base, raw_mismatched)
    matched_target, _ = apply_risk_shift(train_base, raw_matched)
    graph_target, _ = apply_risk_shift(train_base, graph_train)
    base_score = support_score(train_base, truth)
    mismatched_gain = support_score(mismatched_target, truth) - base_score
    matched_gain = support_score(matched_target, truth) - base_score
    graph_gain = support_score(graph_target, truth) - base_score
    graph_advantage = graph_gain - matched_gain

    basic_train, basic_names = basic_features(pos_train, blocked_train)
    basic_test, _ = basic_features(pos_test, blocked_test)
    official_train = np.concatenate(
        [basic_train, geodesic["train_features"], scene["train_features"]], axis=1
    ).astype(np.float32)
    official_test = np.concatenate(
        [basic_test, geodesic["test_features"], scene["test_features"]], axis=1
    ).astype(np.float32)
    official_names = basic_names + geodesic_names + pointcloud_names
    residual_names = [f"residual_prob_{index:02d}" for index in range(96)]
    matched_full_train = np.concatenate([official_train, matched_prob], axis=1)
    matched_full_test = np.concatenate([official_test, test_v129_prob], axis=1)

    domain_variants = {
        "basic": domain_variant("basic", basic_train, basic_test, basic_names),
        "official_without_prob": domain_variant(
            "official", official_train, official_test, official_names
        ),
        "mismatched_prob_only": domain_variant(
            "mismatched_prob",
            train_v125_prob,
            test_v129_prob,
            [f"mismatched_prob_{index:02d}" for index in range(96)],
        ),
        "matched_residual_prob_only": domain_variant(
            "matched_residual_prob", matched_prob, test_v129_prob, residual_names
        ),
        "official_plus_matched_residual_prob": domain_variant(
            "official_matched",
            matched_full_train,
            matched_full_test,
            official_names + residual_names,
        ),
    }

    spatial_groups = __import__("sklearn.cluster", fromlist=["KMeans"]).KMeans(
        n_clusters=25, random_state=RANDOM_STATE, n_init=20
    ).fit_predict(pos_train[:, :2])
    predictability = {
        "official_without_prob": regression_oof(
            official_train, graph_advantage, spatial_groups, valid
        ),
        "official_plus_matched_residual_prob": regression_oof(
            matched_full_train, graph_advantage, spatial_groups, valid
        ),
        "full_with_support_signature": regression_oof(
            np.concatenate(
                [matched_full_train, support_features(train_base)], axis=1
            ),
            graph_advantage,
            spatial_groups,
            valid,
        ),
    }

    report = {
        "method": "v143_matched_residual_probability_analysis",
        "finding": (
            "v143 train graph used v125 propagation-template probabilities while "
            "the test graph used v129 bad-residual-cluster probabilities; rebuild "
            "train residual probabilities in the same 96-class semantic space"
        ),
        "reconstruction": reconstruction,
        "probability_calibration": {
            "train_v125_top": summarize(train_v125_prob.max(axis=1)),
            "train_v125_entropy": summarize(entropy01(train_v125_prob)),
            "train_v129_matched_top": summarize(matched_prob.max(axis=1)),
            "train_v129_matched_entropy": summarize(entropy01(matched_prob)),
            "test_v129_top": summarize(test_v129_prob.max(axis=1)),
            "test_v129_entropy": summarize(entropy01(test_v129_prob)),
        },
        "risk": {
            "train_mismatched": summarize(raw_mismatched),
            "train_matched": summarize(raw_matched),
            "train_matched_graph": summarize(graph_train),
            "test": summarize(raw_test),
            "active_fraction": {
                "train_mismatched": float(np.mean(raw_mismatched > 0.64)),
                "train_matched": float(np.mean(raw_matched > 0.64)),
                "train_matched_graph": float(np.mean(graph_train > 0.64)),
                "test": float(np.mean(raw_test > 0.64)),
            },
        },
        "support_gain": {
            "mismatched_raw": summarize(mismatched_gain[valid]),
            "matched_raw": summarize(matched_gain[valid]),
            "matched_graph": summarize(graph_gain[valid]),
            "matched_graph_advantage": summarize(graph_advantage[valid]),
            "graph_advantage_corr_with_disagreement": safe_corr(
                graph_advantage[valid], disagreement_train[valid]
            ),
        },
        "domain_variants": domain_variants,
        "spatial_oof_graph_advantage": predictability,
        "outputs": {
            "train_matched_residual_probabilities": PATHS.relative(TRAIN_PROB_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {TRAIN_PROB_OUT}")
    print(f"wrote {REPORT}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
