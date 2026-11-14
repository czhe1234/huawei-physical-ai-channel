#!/usr/bin/env python3
"""Augment matched wall paths with v309 reusable effective sources."""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy.spatial import cKDTree

import analyze_v208_strict_historical_replay as v208
import analyze_v264_patch_scatterer_transformer as v264
import analyze_v266_learned_scatterer_joint_projection as v266
import analyze_v273_planar_barrier_kernel as v273
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v308_joint_matched_patch_renderer as v308
import analyze_v309_unmatched_effective_scatterers as v309
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 310
VERSION = "v310_augmented_effective_source_renderer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LABEL_CACHE = OUT_DIR / "v310_augmented_path_labels.npz"
TARGETS_OUT = OUT_DIR / "v310_joint_targets.npz"
OUTPUT = OUT_DIR / "v310_Round1_Test_Channel.npy"
REPORT = REPORT_DIR / f"{VERSION}.json"

OUTER_SEEDS = (310, 1310, 2310, 3310, 4310)
OUTER_COUNT = 320
INNER_COUNT = 240
EPOCHS = 80
K_ANCHOR = 16
MIN_OFFICIAL_JOINT_DELTA = 0.01
FINAL_SEEDS = (31001, 31031, 31061)


def effective_metadata(
    cluster: dict[str, np.ndarray], train_pos: np.ndarray
) -> dict[str, np.ndarray]:
    cluster_ids = cluster["cluster_ids"].astype(np.int32)
    centers = cluster["cluster_centers"].astype(np.float32)
    labels = cluster["labels"].astype(np.int32)
    point_ids = cluster["point_id"].astype(np.int64)
    mass = cluster["mass"].astype(np.float32)
    support_centroid = np.zeros((len(cluster_ids), 2), dtype=np.float32)
    support_radius = np.zeros(len(cluster_ids), dtype=np.float32)
    occurrence = np.zeros(len(cluster_ids), dtype=np.float32)
    mass_mean = np.zeros(len(cluster_ids), dtype=np.float32)
    mass_p90 = np.zeros(len(cluster_ids), dtype=np.float32)
    for row, cluster_id in enumerate(cluster_ids):
        selected = labels == cluster_id
        ue = train_pos[point_ids[selected], :2]
        support_centroid[row] = np.mean(ue, axis=0)
        support_radius[row] = float(
            np.quantile(np.linalg.norm(ue - support_centroid[row], axis=1), 0.90)
        )
        occurrence[row] = float(np.sum(selected))
        mass_mean[row] = float(np.mean(mass[selected]))
        mass_p90[row] = float(np.quantile(mass[selected], 0.90))
    return {
        "cluster_ids": cluster_ids,
        "centers": centers,
        "support_centroid": support_centroid,
        "support_radius": support_radius,
        "occurrence": occurrence,
        "mass_mean": mass_mean,
        "mass_p90": mass_p90,
    }


def extend_features(
    base: v264.FeatureSet,
    pos: np.ndarray,
    metadata: dict[str, np.ndarray],
    wall_patch_count: int,
) -> tuple[v264.FeatureSet, np.ndarray]:
    centers = metadata["centers"].astype(np.float64)
    count = len(pos)
    source_count = len(centers)
    path_dim = base.path_numeric.shape[2]
    numeric = np.zeros((count, source_count, path_dim), dtype=np.float32)
    bs_xy = v304.BS[:2]
    bs_to_source = centers - bs_xy[None, :]
    source_to_ue = pos[:, None, :2] - centers[None, :, :]
    first_length = np.linalg.norm(bs_to_source, axis=1)
    second_length = np.linalg.norm(source_to_ue, axis=2)
    path_length = first_length[None, :] + second_length
    direct = np.linalg.norm(pos[:, :2] - bs_xy[None, :], axis=1)
    excess = path_length - direct[:, None]
    delay = path_length / v304.METERS_PER_BIN
    first_unit = bs_to_source / np.maximum(first_length[:, None], 1e-8)
    second_unit = source_to_ue / np.maximum(second_length[:, :, None], 1e-8)
    source_xyz = np.zeros((source_count, 3), dtype=np.float32)
    source_xyz[:, :2] = centers
    source_xyz[:, 2] = 1.5
    numeric[:, :, 0] = delay / 24.0
    numeric[:, :, 1] = path_length / 500.0
    numeric[:, :, 2] = excess / 300.0
    numeric[:, :, 3] = np.log1p(metadata["occurrence"])[None, :] / 8.0
    numeric[:, :, 4:7] = (source_xyz[None, :, :] - v304.BS[None, None, :]) / 500.0
    numeric[:, :, 7:10] = (source_xyz[None, :, :] - pos[:, None, :]) / 500.0
    numeric[:, :, 10:12] = first_unit[None, :, :]
    numeric[:, :, 13:15] = second_unit
    numeric[:, :, 16] = metadata["support_centroid"][None, :, 0] / 400.0
    numeric[:, :, 17] = metadata["support_centroid"][None, :, 1] / 350.0
    support_distance = np.linalg.norm(
        pos[:, None, :2] - metadata["support_centroid"][None, :, :], axis=2
    )
    numeric[:, :, 18] = support_distance / 300.0
    numeric[:, :, 19] = metadata["support_radius"][None, :] / 300.0
    numeric[:, :, 20] = np.log1p(metadata["occurrence"])[None, :] / 8.0
    numeric[:, :, 21] = metadata["mass_mean"][None, :]
    numeric[:, :, 22] = metadata["mass_p90"][None, :]
    patch_id = np.broadcast_to(
        wall_patch_count + 1 + np.arange(source_count, dtype=np.int64)[None, :],
        (count, source_count),
    ).copy()
    path_type = np.ones((count, source_count), dtype=np.int64)
    path_valid = np.ones((count, source_count), dtype=bool)
    global_features = base.global_features
    extended = v264.FeatureSet(
        global_features=global_features,
        path_numeric=np.concatenate([base.path_numeric, numeric], axis=1),
        patch_id=np.concatenate([base.patch_id, patch_id], axis=1),
        path_type=np.concatenate([base.path_type, path_type], axis=1),
        path_valid=np.concatenate([base.path_valid, path_valid], axis=1),
        delay=np.concatenate([base.delay, delay.astype(np.float32)], axis=1),
    )
    effective_h = np.broadcast_to(
        v304.expected_h_bin(
            np.arctan2(bs_to_source[:, 1], bs_to_source[:, 0])
        )[None, :],
        (count, source_count),
    ).astype(np.float32)
    return extended, effective_h


def build_augmented_labels(
    metadata: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if LABEL_CACHE.is_file():
        data = np.load(LABEL_CACHE)
        return data["labels"], data["quality"], json.loads(str(data["report_json"].item()))
    component = np.load(v297.COMPONENT_CACHE)
    inverse = np.load(v304.SCATTERER_CACHE)
    matches = np.load(v305.MATCH_CACHE)
    clusters = np.load(v309.CLUSTER_CACHE)
    fingerprint = np.load(v300.FINGERPRINT_CACHE)
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]
    source_count = len(metadata["cluster_ids"])
    labels = np.zeros((len(component["delay"]), v306.PATH_COUNT + source_count), dtype=np.float32)
    los_component = inverse["los_component"].astype(np.int64)
    labels[:, 0] = component["weight"][np.arange(len(labels)), los_component]

    good = (
        (matches["length_error_bins"] <= v309.GOOD_LENGTH_ERROR)
        & (matches["h_error_bins"] <= v309.GOOD_H_ERROR)
    )
    match_confidence = np.exp(
        -0.5
        * (
            (matches["length_error_bins"][good] / 0.50) ** 2
            + (matches["h_error_bins"][good] / 1.00) ** 2
        )
    ).astype(np.float32)
    np.add.at(
        labels,
        (
            matches["point_id"][good],
            matches["candidate_id"][good].astype(np.int64) + 1,
        ),
        matches["mass"][good] * match_confidence,
    )

    cluster_ids = clusters["cluster_ids"].astype(np.int32)
    cluster_lookup = {int(value): row for row, value in enumerate(cluster_ids)}
    selected = clusters["labels"] >= 0
    point_id = clusters["point_id"][selected].astype(np.int64)
    component_id = clusters["component_id"][selected].astype(np.int64)
    assigned = clusters["labels"][selected].astype(np.int32)
    center = np.asarray([metadata["centers"][cluster_lookup[int(value)]] for value in assigned])
    observed_length = (
        component["delay"][point_id, component_id] - float(calibration["intercept_bins"])
    ) / float(calibration["slope_bins_per_meter"])
    observed_h = v304.horizontal_center(fingerprint["horizontal"])[point_id, component_id]
    predicted_length = (
        np.linalg.norm(center - v304.BS[:2], axis=1)
        + np.linalg.norm(center - np.load(PATHS.round1 / "Round1_Train_Pos.npy")[point_id, :2], axis=1)
    )
    predicted_h = v304.expected_h_bin(
        np.arctan2(center[:, 1] - v304.BS[1], center[:, 0] - v304.BS[0])
    )
    length_error = np.abs(predicted_length - observed_length) / v304.METERS_PER_BIN
    h_error = v305.circular_distance(predicted_h, observed_h, 16.0)
    confidence = np.exp(
        -0.5 * ((length_error / 0.50) ** 2 + (h_error / 1.00) ** 2)
    ).astype(np.float32)
    destination = np.asarray(
        [v306.PATH_COUNT + cluster_lookup[int(value)] for value in assigned], dtype=np.int64
    )
    np.add.at(
        labels,
        (point_id, destination),
        component["weight"][point_id, component_id] * confidence,
    )
    represented = labels.sum(axis=1)
    quality = np.clip(represented, 0.0, 1.0).astype(np.float32)
    valid = represented > 1e-12
    labels[valid] /= represented[valid, None]
    labels[~valid, 0] = 1.0
    effective_mass = labels[:, v306.PATH_COUNT :].sum(axis=1)
    report = {
        "represented_mass": summarize(represented),
        "quality": summarize(quality),
        "active_candidate_count": summarize((labels > 1e-5).sum(axis=1)),
        "direct_mass": summarize(labels[:, 0]),
        "effective_source_mass": summarize(effective_mass),
        "effective_labeled_points": int(np.sum(effective_mass > 1e-5)),
        "candidate_count": int(labels.shape[1]),
    }
    np.savez_compressed(
        LABEL_CACHE,
        labels=labels,
        quality=quality,
        report_json=np.asarray(json.dumps(report)),
    )
    return labels, quality, report


def test_anchors(
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    valid_ids: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(train_pos[valid_ids, :2])
    distance, local = tree.query(test_pos[:, :2], k=K_ANCHOR)
    weight = 1.0 / np.maximum(distance, 0.05) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    sh = v297.normalize_rows(
        np.sum(weight[:, :, None, None] * truth_sh[valid_ids[local]], axis=1, dtype=np.float64)
    )
    d = v297.normalize_rows(
        np.sum(weight[:, :, None] * truth_d[valid_ids[local]], axis=1, dtype=np.float64)
    )
    return sh, d


def main() -> None:
    cluster_data = np.load(v309.CLUSTER_CACHE)
    cluster = {key: cluster_data[key] for key in cluster_data.files}
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    metadata = effective_metadata(cluster, train_pos)
    scene = np.load(v306.SCENE_CACHE)
    patches = scene["wall_patches"].astype(np.float32)
    train_candidate_data = np.load(v306.TRAIN_CANDIDATES)
    train_candidates = {key: train_candidate_data[key] for key in train_candidate_data.files}
    test_candidate_data = np.load(v306.TEST_CANDIDATES)
    test_candidates = {key: test_candidate_data[key] for key in test_candidate_data.files}
    train_base, test_base = v306.make_features(
        train_pos, test_pos, train_candidates, test_candidates, patches
    )
    train_raw, train_effective_h = extend_features(
        train_base, train_pos, metadata, len(patches)
    )
    test_raw, test_effective_h = extend_features(
        test_base, test_pos, metadata, len(patches)
    )
    train_h = np.concatenate(
        [v308.candidate_h_center(train_pos, train_candidates), train_effective_h], axis=1
    )
    test_h = np.concatenate(
        [v308.candidate_h_center(test_pos, test_candidates), test_effective_h], axis=1
    )
    labels, quality, label_report = build_augmented_labels(metadata)
    targets = np.load(v306.TARGET_CACHE)
    truth_sh = targets["sh"].astype(np.float32)
    truth_d = targets["d"].astype(np.float32)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_count = len(patches) + len(metadata["centers"])

    fold_rows = []
    base_sh_parts = []
    model_sh_parts = []
    truth_sh_parts = []
    base_d_parts = []
    model_d_parts = []
    truth_d_parts = []
    selected_epochs = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v308.v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = v273.matched_subset(
            outer_fit, train_pos[:, :2], test_pos[:, :2], INNER_COUNT, seed + 10_000
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)
        inner_fit_raw = train_raw.take(inner_fit)
        inner_val_raw = train_raw.take(inner_ids)
        inner_fit_feature, _full, inner_val_feature, _scale = v264.standardize_features(
            inner_fit_raw, train_raw, inner_val_raw
        )
        inner_anchor_sh = v308.generic_knn_anchor(inner_fit, inner_fit, train_pos, truth_sh)
        inner_anchor_d = v308.generic_knn_anchor(inner_fit, inner_fit, train_pos, truth_d)
        inner_val_anchor_sh = v308.generic_knn_anchor(inner_ids, inner_fit, train_pos, truth_sh)
        inner_val_anchor_d = v308.generic_knn_anchor(inner_ids, inner_fit, train_pos, truth_d)
        _model, inner_report = v308.fit_model(
            inner_fit_feature,
            train_h[inner_fit],
            inner_anchor_sh,
            inner_anchor_d,
            truth_sh[inner_fit],
            truth_d[inner_fit],
            labels[inner_fit],
            quality[inner_fit],
            seed,
            EPOCHS,
            patch_count,
            device,
            (
                inner_val_feature,
                train_h[inner_ids],
                inner_val_anchor_sh,
                inner_val_anchor_d,
                truth_sh[inner_ids],
                truth_d[inner_ids],
            ),
        )
        selected_epoch = int(inner_report["best_epoch"])
        selected_epochs.append(selected_epoch)
        outer_fit_raw = train_raw.take(outer_fit)
        outer_val_raw = train_raw.take(outer_ids)
        outer_fit_feature, _full, outer_val_feature, _scale = v264.standardize_features(
            outer_fit_raw, train_raw, outer_val_raw
        )
        outer_anchor_sh = v308.generic_knn_anchor(outer_fit, outer_fit, train_pos, truth_sh)
        outer_anchor_d = v308.generic_knn_anchor(outer_fit, outer_fit, train_pos, truth_d)
        outer_val_anchor_sh = v308.generic_knn_anchor(outer_ids, outer_fit, train_pos, truth_sh)
        outer_val_anchor_d = v308.generic_knn_anchor(outer_ids, outer_fit, train_pos, truth_d)
        if selected_epoch == 0:
            pred_sh = outer_val_anchor_sh.copy()
            pred_d = outer_val_anchor_d.copy()
            gate = np.zeros(len(outer_ids), dtype=np.float32)
            outer_training = {"best_epoch": 0}
        else:
            model, outer_training = v308.fit_model(
                outer_fit_feature,
                train_h[outer_fit],
                outer_anchor_sh,
                outer_anchor_d,
                truth_sh[outer_fit],
                truth_d[outer_fit],
                labels[outer_fit],
                quality[outer_fit],
                seed + 50_000,
                selected_epoch,
                patch_count,
                device,
            )
            pred_sh, pred_d, gate = v308.predict_joint(
                model, outer_val_feature, train_h[outer_ids],
                outer_val_anchor_d, outer_val_anchor_sh, device
            )
        base_sh_score = float(np.mean(v297.row_cos(outer_val_anchor_sh, truth_sh[outer_ids])))
        base_d_score = float(np.mean(v297.row_cos(outer_val_anchor_d, truth_d[outer_ids])))
        model_sh_score = float(np.mean(v297.row_cos(pred_sh, truth_sh[outer_ids])))
        model_d_score = float(np.mean(v297.row_cos(pred_d, truth_d[outer_ids])))
        base_sh_parts.append(outer_val_anchor_sh)
        model_sh_parts.append(pred_sh)
        truth_sh_parts.append(truth_sh[outer_ids])
        base_d_parts.append(outer_val_anchor_d)
        model_d_parts.append(pred_d)
        truth_d_parts.append(truth_d[outer_ids])
        fold_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "selected_epoch": selected_epoch,
                "base_sh": base_sh_score,
                "base_d": base_d_score,
                "model_sh": model_sh_score,
                "model_d": model_d_score,
                "raw_weighted_delta": 0.52 * (model_sh_score - base_sh_score) + 0.48 * (model_d_score - base_d_score),
                "gate": summarize(gate),
                "inner": inner_report,
                "outer_training": outer_training,
            }
        )
        print(
            f"v310 fold {fold + 1}/5 epoch={selected_epoch} "
            f"d={model_d_score-base_d_score:+.6f} sh={model_sh_score-base_sh_score:+.6f} "
            f"weighted={fold_rows[-1]['raw_weighted_delta']:+.6f}", flush=True
        )

    base_sh_all = np.concatenate(base_sh_parts)
    model_sh_all = np.concatenate(model_sh_parts)
    truth_sh_all = np.concatenate(truth_sh_parts)
    base_d_all = np.concatenate(base_d_parts)
    model_d_all = np.concatenate(model_d_parts)
    truth_d_all = np.concatenate(truth_d_parts)
    selected, alpha_rows = v308.choose_alphas(
        base_sh_all, model_sh_all, truth_sh_all,
        base_d_all, model_d_all, truth_d_all,
    )
    blended_sh = v297.normalize_rows(
        base_sh_all + selected["alpha_sh"] * (model_sh_all - base_sh_all)
    )
    blended_d = v297.normalize_rows(
        base_d_all + selected["alpha_d"] * (model_d_all - base_d_all)
    )
    base_sh_score = float(np.mean(v297.row_cos(base_sh_all, truth_sh_all)))
    base_d_score = float(np.mean(v297.row_cos(base_d_all, truth_d_all)))
    blended_sh_score = float(np.mean(v297.row_cos(blended_sh, truth_sh_all)))
    blended_d_score = float(np.mean(v297.row_cos(blended_d, truth_d_all)))
    base_weighted = 0.52 * base_sh_score + 0.48 * base_d_score
    blended_weighted = 0.52 * blended_sh_score + 0.48 * blended_d_score
    fold_deltas = []
    start = 0
    for row, base_sh, model_sh, truth_sh_fold, base_d, model_d, truth_d_fold in zip(
        fold_rows, base_sh_parts, model_sh_parts, truth_sh_parts,
        base_d_parts, model_d_parts, truth_d_parts,
    ):
        end = start + len(base_sh)
        current_sh = blended_sh[start:end]
        current_d = blended_d[start:end]
        sh_score = float(np.mean(v297.row_cos(current_sh, truth_sh_fold)))
        d_score = float(np.mean(v297.row_cos(current_d, truth_d_fold)))
        weighted = 0.52 * sh_score + 0.48 * d_score
        base = 0.52 * float(row["base_sh"]) + 0.48 * float(row["base_d"])
        row["blended_sh"] = sh_score
        row["blended_d"] = d_score
        row["blended_weighted_delta"] = weighted - base
        fold_deltas.append(weighted - base)
        start = end
    official_delta = blended_weighted - base_weighted
    official_selection = bool(
        official_delta >= MIN_OFFICIAL_JOINT_DELTA
        and sum(value > 0.0 for value in fold_deltas) >= 4
    )

    deployment = None
    if official_selection:
        full_raw = train_raw.take(valid_ids)
        full_feature, _unused, test_feature, _scale = v264.standardize_features(
            full_raw, train_raw, test_raw
        )
        train_anchor_sh = v308.generic_knn_anchor(valid_ids, valid_ids, train_pos, truth_sh)
        train_anchor_d = v308.generic_knn_anchor(valid_ids, valid_ids, train_pos, truth_d)
        test_anchor_sh, test_anchor_d = test_anchors(
            train_pos, test_pos, valid_ids, truth_sh, truth_d
        )
        final_epochs = max(1, int(np.median([value for value in selected_epochs if value > 0])))
        sh_predictions = []
        d_predictions = []
        checkpoints = []
        for seed in FINAL_SEEDS:
            model, _training = v308.fit_model(
                full_feature,
                train_h[valid_ids],
                train_anchor_sh,
                train_anchor_d,
                truth_sh[valid_ids],
                truth_d[valid_ids],
                labels[valid_ids],
                quality[valid_ids],
                seed,
                final_epochs,
                patch_count,
                device,
            )
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(model.state_dict(), checkpoint)
            checkpoints.append(PATHS.relative(checkpoint))
            pred_sh, pred_d, _gate = v308.predict_joint(
                model, test_feature, test_h, test_anchor_d, test_anchor_sh, device
            )
            sh_predictions.append(pred_sh)
            d_predictions.append(pred_d)
        model_test_sh = v297.normalize_rows(np.mean(sh_predictions, axis=0))
        model_test_d = v297.normalize_rows(np.mean(d_predictions, axis=0))
        target_sh = v297.normalize_rows(
            test_anchor_sh + selected["alpha_sh"] * (model_test_sh - test_anchor_sh)
        )
        target_d = v297.normalize_rows(
            test_anchor_d + selected["alpha_d"] * (model_test_d - test_anchor_d)
        )
        np.savez_compressed(TARGETS_OUT, sh=target_sh, d=target_d)
        diagnostics = v266.alternating_project(v306.BASE, OUTPUT, target_sh, target_d)
        deployment = {
            "final_epochs": final_epochs,
            "checkpoints": checkpoints,
            "targets": PATHS.relative(TARGETS_OUT),
            "prediction": PATHS.relative(OUTPUT),
            "local_score": v208.score_prediction(OUTPUT),
            "diagnostics": {key: summarize(value) for key, value in diagnostics.items()},
        }

    report = {
        "method": VERSION,
        "status": "official_selection_passed" if official_selection else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Reusable v309 effective sources add the stable multi-hop/diffraction path families "
            "missing from v308 one-bounce official wall candidates."
        ),
        "effective_source_metadata": {
            "count": int(len(metadata["centers"])),
            "occurrence": summarize(metadata["occurrence"]),
            "support_radius": summarize(metadata["support_radius"]),
        },
        "label_report": label_report,
        "fold_rows": fold_rows,
        "oof": {
            "base_sh": base_sh_score,
            "base_d": base_d_score,
            "base_weighted": base_weighted,
            "selected": selected,
            "blended_sh": blended_sh_score,
            "blended_d": blended_d_score,
            "blended_weighted": blended_weighted,
            "delta": official_delta,
            "fold_deltas": fold_deltas,
            "fold_wins": int(sum(value > 0.0 for value in fold_deltas)),
            "minimum_required_delta": MIN_OFFICIAL_JOINT_DELTA,
            "official_selection": official_selection,
            "alpha_grid": alpha_rows,
        },
        "deployment": deployment,
        "outputs": {
            "labels": PATHS.relative(LABEL_CACHE),
            "report": PATHS.relative(REPORT),
            "targets": PATHS.relative(TARGETS_OUT) if TARGETS_OUT.is_file() else None,
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["label_report"], indent=2), flush=True)
    print(json.dumps(report["oof"], indent=2), flush=True)


if __name__ == "__main__":
    main()
