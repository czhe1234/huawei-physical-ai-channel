#!/usr/bin/env python3
"""Joint renderer supervised by identity-matched high-depth geometry paths."""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy.spatial import cKDTree

import analyze_v264_patch_scatterer_transformer as v264
import analyze_v273_planar_barrier_kernel as v273
import analyze_v293_interaction_physical_path_law as v293
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v308_joint_matched_patch_renderer as v308
import analyze_v310_augmented_effective_source_renderer as v310
import analyze_v312_highdepth_path_identity_match as v312
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 313
VERSION = "v313_highdepth_matched_path_renderer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LABEL_CACHE = OUT_DIR / "v313_matched_path_labels.npz"
TARGETS_OUT = OUT_DIR / "v313_joint_targets.npz"
OUTPUT = OUT_DIR / "v313_Round1_Test_Channel.npy"
REPORT = REPORT_DIR / f"{VERSION}.json"

PATH_LIMIT = 64
OUTER_SEEDS = (313, 1313, 2313, 3313, 4313)
OUTER_COUNT = 320
INNER_COUNT = 240
EPOCHS = 80
K_ANCHOR = 16
MIN_OFFICIAL_JOINT_DELTA = 0.01
FINAL_SEEDS = (31301, 31331, 31361)


def map_first_patch(first: np.ndarray, valid: np.ndarray, patches: np.ndarray) -> np.ndarray:
    normal = patches[:, :2].astype(np.float64)
    tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
    midpoint_t = 0.5 * (patches[:, 3] + patches[:, 4])
    midpoint = patches[:, 2, None] * normal + midpoint_t[:, None] * tangent
    tree = cKDTree(midpoint)
    flat = first.reshape(-1, 3).astype(np.float64)
    take = min(64, len(patches))
    nearby = tree.query(flat[:, :2], k=take)[1]
    selected_patch = patches[nearby]
    selected_normal = selected_patch[:, :, :2]
    selected_tangent = np.stack(
        [-selected_normal[:, :, 1], selected_normal[:, :, 0]], axis=2
    )
    perpendicular = np.abs(
        np.sum(selected_normal * flat[:, None, :2], axis=2) - selected_patch[:, :, 2]
    )
    coordinate = np.sum(selected_tangent * flat[:, None, :2], axis=2)
    along = np.maximum(selected_patch[:, :, 3] - coordinate, 0.0) + np.maximum(
        coordinate - selected_patch[:, :, 4], 0.0
    )
    vertical = np.maximum(selected_patch[:, :, 5] - flat[:, None, 2], 0.0) + np.maximum(
        flat[:, None, 2] - selected_patch[:, :, 6], 0.0
    )
    cost = perpendicular + along + 0.25 * vertical
    choice = np.argmin(cost, axis=1)
    patch_id = nearby[np.arange(len(flat)), choice].astype(np.int64) + 1
    best_cost = cost[np.arange(len(flat)), choice]
    usable = valid.reshape(-1) & (flat[:, 2] > 0.4) & (best_cost < 8.0)
    patch_id[~usable] = 0
    return patch_id.reshape(valid.shape)


def highdepth_geometry(
    path_cache: dict[str, np.ndarray],
    interaction_cache: dict[str, np.ndarray],
    prefix: str,
    pos: np.ndarray,
    patches: np.ndarray,
) -> dict[str, np.ndarray]:
    valid = (
        path_cache[f"{prefix}_valid"][:, :PATH_LIMIT].astype(bool)
        & (path_cache[f"{prefix}_interaction_count"][:, :PATH_LIMIT] >= 2)
    )
    length = path_cache[f"{prefix}_path_length"][:, :PATH_LIMIT].astype(np.float32)
    depth = path_cache[f"{prefix}_interaction_count"][:, :PATH_LIMIT].astype(np.int64)
    vertices = np.nan_to_num(
        path_cache[f"{prefix}_vertices"][:, :PATH_LIMIT].astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    first = vertices[:, :, 0].copy()
    last = np.zeros_like(first)
    for step in range(16):
        last = np.where((depth == step + 1)[..., None], vertices[:, :, step], last)
    bs = v304.BS.astype(np.float32)
    first_vector = first - bs[None, None, :]
    last_vector = pos[:, None, :].astype(np.float32) - last
    first_unit = first_vector / np.maximum(np.linalg.norm(first_vector, axis=2, keepdims=True), 1e-6)
    last_unit = last_vector / np.maximum(np.linalg.norm(last_vector, axis=2, keepdims=True), 1e-6)
    direct = np.linalg.norm(pos - v304.BS[None, :], axis=1).astype(np.float32)
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]
    delay = (
        float(calibration["intercept_bins"])
        + float(calibration["slope_bins_per_meter"]) * length
    ).astype(np.float32)
    delay[~valid] = 0.0
    score = np.nan_to_num(
        path_cache[f"{prefix}_score"][:, :PATH_LIMIT].astype(np.float32),
        nan=-20.0,
        neginf=-20.0,
        posinf=8.0,
    )
    base_dim = 26
    base_numeric = np.zeros((len(pos), PATH_LIMIT, base_dim), dtype=np.float32)
    base_numeric[:, :, 0] = delay / 24.0
    base_numeric[:, :, 1] = length / 500.0
    base_numeric[:, :, 2] = (length - direct[:, None]) / 300.0
    base_numeric[:, :, 3] = np.clip(score, -20.0, 8.0) / 8.0
    base_numeric[:, :, 4:7] = first_vector / 500.0
    base_numeric[:, :, 7:10] = last_vector / 500.0
    base_numeric[:, :, 10:13] = first_unit
    base_numeric[:, :, 13:16] = last_unit
    patch_id = map_first_patch(first, valid, patches)
    safe_patch = np.clip(patch_id - 1, 0, len(patches) - 1)
    selected_patch = patches[safe_patch]
    patch_numeric = np.zeros((len(pos), PATH_LIMIT, 10), dtype=np.float32)
    patch_numeric[:, :, 0:2] = selected_patch[:, :, 0:2]
    patch_numeric[:, :, 2] = selected_patch[:, :, 2] / 500.0
    patch_numeric[:, :, 3] = (selected_patch[:, :, 4] - selected_patch[:, :, 3]) / 250.0
    patch_numeric[:, :, 4] = (selected_patch[:, :, 5] + selected_patch[:, :, 6]) / 60.0
    patch_numeric[:, :, 5] = (selected_patch[:, :, 6] - selected_patch[:, :, 5]) / 40.0
    patch_numeric[:, :, 6] = np.log1p(selected_patch[:, :, 7]) / 10.0
    patch_numeric[:, :, 7] = selected_patch[:, :, 8]
    patch_numeric[:, :, 8] = selected_patch[:, :, 9]
    patch_numeric[:, :, 9] = selected_patch[:, :, 11] / 160.0
    patch_numeric[patch_id == 0] = 0.0
    base_numeric[:, :, 16:26] = patch_numeric
    physical = interaction_cache[f"{prefix}_path_features"][:, :PATH_LIMIT].astype(np.float32)
    numeric = np.concatenate([base_numeric, physical], axis=2)
    numeric[~valid] = 0.0
    h = v305.expected_h_bin(first)
    h[~valid] = 0.0
    return {
        "numeric": numeric,
        "patch_id": patch_id,
        "valid": valid,
        "delay": delay,
        "h": h.astype(np.float32),
    }


def extend_features(base: v264.FeatureSet, high: dict[str, np.ndarray]) -> v264.FeatureSet:
    extra_dim = high["numeric"].shape[2] - base.path_numeric.shape[2]
    base_numeric = np.concatenate(
        [base.path_numeric, np.zeros(base.path_numeric.shape[:2] + (extra_dim,), dtype=np.float32)],
        axis=2,
    )
    return v264.FeatureSet(
        global_features=base.global_features,
        path_numeric=np.concatenate([base_numeric, high["numeric"]], axis=1),
        patch_id=np.concatenate([base.patch_id, high["patch_id"]], axis=1),
        path_type=np.concatenate(
            [base.path_type, np.ones(high["valid"].shape, dtype=np.int64)], axis=1
        ),
        path_valid=np.concatenate([base.path_valid, high["valid"]], axis=1),
        delay=np.concatenate([base.delay, high["delay"]], axis=1),
    )


def build_labels() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if LABEL_CACHE.is_file():
        data = np.load(LABEL_CACHE)
        return data["labels"], data["quality"], json.loads(str(data["report_json"].item()))
    base_labels, base_quality, _base_report = v306.build_labels()
    raw = base_labels * base_quality[:, None]
    labels = np.concatenate(
        [raw, np.zeros((len(raw), PATH_LIMIT), dtype=np.float32)], axis=1
    )
    match = np.load(v312.MATCH_CACHE)
    good = (
        (match["length_error_bins"] <= v312.GOOD_LENGTH_ERROR)
        & (match["h_error_bins"] <= v312.GOOD_H_ERROR)
        & (match["candidate_id"] < PATH_LIMIT)
    )
    confidence = np.exp(-0.5 * match["cost"][good]).astype(np.float32)
    np.add.at(
        labels,
        (
            match["point_id"][good].astype(np.int64),
            v306.PATH_COUNT + match["candidate_id"][good].astype(np.int64),
        ),
        match["mass"][good] * confidence,
    )
    represented = labels.sum(axis=1)
    quality = np.clip(represented, 0.0, 1.0).astype(np.float32)
    valid = represented > 1e-12
    labels[valid] /= represented[valid, None]
    labels[~valid, 0] = 1.0
    high_mass = labels[:, v306.PATH_COUNT :].sum(axis=1)
    report = {
        "candidate_count": int(labels.shape[1]),
        "represented_mass": summarize(represented),
        "quality": summarize(quality),
        "active_candidate_count": summarize((labels > 1e-5).sum(axis=1)),
        "direct_mass": summarize(labels[:, 0]),
        "highdepth_mass": summarize(high_mass),
        "highdepth_labeled_points": int(np.sum(high_mass > 1e-5)),
        "highdepth_good_match_count": int(np.sum(good)),
    }
    np.savez_compressed(
        LABEL_CACHE,
        labels=labels,
        quality=quality,
        report_json=np.asarray(json.dumps(report)),
    )
    return labels, quality, report


def main() -> None:
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    scene = np.load(v306.SCENE_CACHE)
    patches = scene["wall_patches"].astype(np.float32)
    train_candidate_data = np.load(v306.TRAIN_CANDIDATES)
    train_candidates = {key: train_candidate_data[key] for key in train_candidate_data.files}
    test_candidate_data = np.load(v306.TEST_CANDIDATES)
    test_candidates = {key: test_candidate_data[key] for key in test_candidate_data.files}
    train_base, test_base = v306.make_features(
        train_pos, test_pos, train_candidates, test_candidates, patches
    )
    path_data = np.load(v312.PATH_CACHE)
    path_cache = {key: path_data[key] for key in path_data.files}
    interaction_data = np.load(v293.FEATURE_CACHE)
    interaction_cache = {key: interaction_data[key] for key in interaction_data.files}
    train_high = highdepth_geometry(path_cache, interaction_cache, "train", train_pos, patches)
    test_high = highdepth_geometry(path_cache, interaction_cache, "test", test_pos, patches)
    train_raw = extend_features(train_base, train_high)
    test_raw = extend_features(test_base, test_high)
    train_h = np.concatenate(
        [v308.candidate_h_center(train_pos, train_candidates), train_high["h"]], axis=1
    )
    test_h = np.concatenate(
        [v308.candidate_h_center(test_pos, test_candidates), test_high["h"]], axis=1
    )
    labels, quality, label_report = build_labels()
    targets = np.load(v306.TARGET_CACHE)
    truth_sh = targets["sh"].astype(np.float32)
    truth_d = targets["d"].astype(np.float32)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_count = len(patches)

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
        inner_fit_feature, _full, inner_val_feature, _scale = v264.standardize_features(
            train_raw.take(inner_fit), train_raw, train_raw.take(inner_ids)
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
        outer_fit_feature, _full, outer_val_feature, _scale = v264.standardize_features(
            train_raw.take(outer_fit), train_raw, train_raw.take(outer_ids)
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
                model,
                outer_val_feature,
                train_h[outer_ids],
                outer_val_anchor_d,
                outer_val_anchor_sh,
                device,
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
                "raw_weighted_delta": 0.52 * (model_sh_score - base_sh_score)
                + 0.48 * (model_d_score - base_d_score),
                "gate": summarize(gate),
                "inner": inner_report,
                "outer_training": outer_training,
            }
        )
        print(
            f"v313 fold {fold + 1}/5 epoch={selected_epoch} "
            f"d={model_d_score-base_d_score:+.6f} sh={model_sh_score-base_sh_score:+.6f} "
            f"weighted={fold_rows[-1]['raw_weighted_delta']:+.6f}",
            flush=True,
        )

    base_sh_all = np.concatenate(base_sh_parts)
    model_sh_all = np.concatenate(model_sh_parts)
    truth_sh_all = np.concatenate(truth_sh_parts)
    base_d_all = np.concatenate(base_d_parts)
    model_d_all = np.concatenate(model_d_parts)
    truth_d_all = np.concatenate(truth_d_parts)
    selected, alpha_rows = v308.choose_alphas(
        base_sh_all,
        model_sh_all,
        truth_sh_all,
        base_d_all,
        model_d_all,
        truth_d_all,
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
    for row, base_sh, truth_sh_fold, base_d, truth_d_fold in zip(
        fold_rows, base_sh_parts, truth_sh_parts, base_d_parts, truth_d_parts
    ):
        end = start + len(base_sh)
        sh_score = float(np.mean(v297.row_cos(blended_sh[start:end], truth_sh_fold)))
        d_score = float(np.mean(v297.row_cos(blended_d[start:end], truth_d_fold)))
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
        full_feature, _unused, test_feature, _scale = v264.standardize_features(
            train_raw.take(valid_ids), train_raw, test_raw
        )
        train_anchor_sh = v308.generic_knn_anchor(valid_ids, valid_ids, train_pos, truth_sh)
        train_anchor_d = v308.generic_knn_anchor(valid_ids, valid_ids, train_pos, truth_d)
        test_anchor_sh, test_anchor_d = v310.test_anchors(
            train_pos, test_pos, valid_ids, truth_sh, truth_d
        )
        positive_epochs = [value for value in selected_epochs if value > 0]
        final_epochs = max(1, int(np.median(positive_epochs)))
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
        diagnostics = v308.v266.alternating_project(v306.BASE, OUTPUT, target_sh, target_d)
        deployment = {
            "final_epochs": final_epochs,
            "checkpoints": checkpoints,
            "targets": PATHS.relative(TARGETS_OUT),
            "prediction": PATHS.relative(OUTPUT),
            "local_score": v308.v208.score_prediction(OUTPUT),
            "diagnostics": {key: summarize(value) for key, value in diagnostics.items()},
        }

    report = {
        "method": VERSION,
        "status": "official_selection_passed" if official_selection else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "v312 identity-matched depth>=2 geometry paths provide explicit high-order "
            "existence/energy supervision missing from v308 one-bounce paths."
        ),
        "path_limit": PATH_LIMIT,
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
