#!/usr/bin/env python3
"""Learn joint geometry residuals on a strict v239-like cross-fit current."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v264_patch_scatterer_transformer as v264
import analyze_v274_planar_barrier_attention as v274
import analyze_v279_joint_component_trajectory as v279
import analyze_v281_v239_crossfit_pdp_current as v281
import analyze_v283_joint_geometry_path_renderer as v283
import analyze_v284_planar_geometry_residual_field as v284
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 285
VERSION = "v285_v239_isomorphic_joint_residual"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
SAMPLES_OUT = OUT_DIR / "v285_v239_crossfit_joint_current.npz"
TARGETS_OUT = OUT_DIR / "v285_joint_targets.npz"
OUTPUT = OUT_DIR / "v285_Round1_Test_Channel.npy"
V281_CACHE = PATHS.cache_dir(281) / "v281_v239_crossfit_pdp_current.npz"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"

LOCAL_GATE = 0.645604
FOLD_COUNT = 5
MAX_EPOCHS = 30
FINAL_SEEDS = (28501, 28531, 28561)
MIN_OFFICIAL_DELTA = 5e-4


def assign_unique_rows(
    ids: np.ndarray, folds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    by_id: dict[int, list[int]] = defaultdict(list)
    for row, point_id in enumerate(ids):
        by_id[int(point_id)].append(row)
    fold_count = np.zeros(FOLD_COUNT, dtype=np.int64)
    selected: list[int] = []
    for point_id, rows in sorted(
        by_id.items(), key=lambda item: (len(item[1]), item[0])
    ):
        best = min(rows, key=lambda row: (fold_count[int(folds[row])], int(folds[row])))
        selected.append(best)
        fold_count[int(folds[best])] += 1
    selected_array = np.asarray(selected, dtype=np.int64)
    order = np.argsort(ids[selected_array])
    selected_array = selected_array[order]
    return selected_array, folds[selected_array].astype(np.int16), {
        "source_rows": int(len(ids)),
        "unique_points": int(len(selected_array)),
        "removed_duplicate_rows": int(len(ids) - len(selected_array)),
        "assigned_fold_count": fold_count.tolist(),
    }


def support_to_sh(support: np.ndarray) -> np.ndarray:
    if support.ndim != 4:
        raise ValueError(f"expected H-V-D support, got {support.shape}")
    if support.shape[1:] != (v279.H_SIZE, 8, v279.S_SIZE):
        raise ValueError(f"unexpected support axes {support.shape}")
    reverse = (-np.arange(v279.S_SIZE)) % v279.S_SIZE
    hd = np.sum(support, axis=2, dtype=np.float64).astype(np.float32)
    return v279.normalize(np.transpose(hd[:, :, reverse], (0, 2, 1)))


def force_d(sh: np.ndarray, target_d: np.ndarray) -> np.ndarray:
    reverse = (-np.arange(v279.S_SIZE)) % v279.S_SIZE
    desired_s = target_d[:, reverse]
    current_s = np.sum(sh, axis=2, dtype=np.float64).astype(np.float32)
    ratio = desired_s / np.maximum(current_s, 1e-12)
    return v279.normalize(sh * ratio[:, :, None])


def build_crossfit_samples(
    truth_sh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    cache = np.load(V281_CACHE)
    all_ids = cache["ids"].astype(np.int64)
    all_folds = cache["fold"].astype(np.int16)
    selected_rows, assigned_folds, assignment_report = assign_unique_rows(
        all_ids, all_folds
    )
    query_ids = all_ids[selected_rows]
    current_d = cache["current_d"][selected_rows].astype(np.float32)
    cached_anchor_d = cache["anchor_d"][selected_rows].astype(np.float32)

    _clean, _raw, _power, valid_delay = v281.v215.load_targets()
    support, valid_support, log_power = v281.v238.build_raw_support()
    valid_ids = np.flatnonzero(valid_delay & valid_support).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    geom = np.load(v281.v238.GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)

    joint_current = np.empty(
        (len(query_ids), v279.S_SIZE, v279.H_SIZE), dtype=np.float32
    )
    reconstructed_anchor = np.empty_like(joint_current)
    anchor_l1_parts: list[np.ndarray] = []
    current_l1_parts: list[np.ndarray] = []
    for fold in range(FOLD_COUNT):
        sample_rows = np.flatnonzero(assigned_folds == fold)
        fold_validation_ids = np.unique(all_ids[all_folds == fold])
        fit_ids = np.setdiff1d(valid_ids, fold_validation_ids)
        query = query_ids[sample_rows]
        anchor_support, _anchor_power, _weights = v281.make_anchor(
            query,
            fit_ids,
            support,
            log_power,
            train_pos,
            train_blocked,
            False,
        )
        anchor_sh = support_to_sh(anchor_support)
        reconstructed_anchor[sample_rows] = anchor_sh
        reconstructed_d = v279.derive_d_numpy(anchor_sh)
        anchor_l1_parts.append(
            np.sum(
                np.abs(reconstructed_d - cached_anchor_d[sample_rows]), axis=1
            ).astype(np.float32)
        )
        current_sh = force_d(anchor_sh, current_d[sample_rows])
        joint_current[sample_rows] = current_sh
        current_l1_parts.append(
            np.sum(
                np.abs(v279.derive_d_numpy(current_sh) - current_d[sample_rows]),
                axis=1,
            ).astype(np.float32)
        )
        print(
            f"v285 reconstruct fold={fold} points={len(sample_rows)} fit={len(fit_ids)}",
            flush=True,
        )
        del anchor_support, anchor_sh, current_sh

    baseline = v279.metrics(
        joint_current, truth_sh[query_ids], v279.derive_d_numpy(truth_sh[query_ids])
    )
    report = {
        **assignment_report,
        "query_id_min": int(np.min(query_ids)),
        "query_id_max": int(np.max(query_ids)),
        "reconstructed_anchor_vs_cached_d_l1": summarize(
            np.concatenate(anchor_l1_parts)
        ),
        "forced_current_vs_cached_d_l1": summarize(
            np.concatenate(current_l1_parts)
        ),
        "joint_current_baseline": baseline,
    }
    np.savez_compressed(
        SAMPLES_OUT,
        query_ids=query_ids,
        assigned_fold=assigned_folds,
        reconstructed_anchor_sh=reconstructed_anchor,
        current_sh=joint_current,
        current_d=current_d,
        cached_anchor_d=cached_anchor_d,
    )
    return query_ids, assigned_folds, joint_current, report


def main() -> None:
    for path in (
        V281_CACHE,
        BASE,
        v283.TARGET_CACHE,
        v283.TRAIN_CANDIDATES,
        v283.TEST_CANDIDATES,
        v283.SCENE_CACHE,
        v283.V265_REPORT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    target_data = np.load(v283.TARGET_CACHE)
    truth_sh = v279.normalize(target_data["sh"].astype(np.float32))
    truth_d = v279.normalize(target_data["d"].astype(np.float32))
    query_ids, assigned_fold, current_sh, sample_report = build_crossfit_samples(
        truth_sh
    )

    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    patches = np.load(v283.SCENE_CACHE)["wall_patches"].astype(np.float32)
    train_candidates = dict(np.load(v283.TRAIN_CANDIDATES))
    test_candidates = dict(np.load(v283.TEST_CANDIDATES))
    train_node, test_node = v257.build_node_features(
        train_pos[:, :2], test_pos[:, :2]
    )
    train_raw = v264.build_features(
        train_pos, train_candidates, patches, train_node
    )
    test_raw = v264.build_features(test_pos, test_candidates, patches, test_node)
    with v283.V265_REPORT.open("r", encoding="utf-8") as handle:
        center_spec = json.load(handle)["selection"]["best"]
    train_s, train_h = v283.centers(
        train_pos, train_candidates, train_raw, center_spec
    )
    test_s, test_h = v283.centers(
        test_pos, test_candidates, test_raw, center_spec
    )

    v283.JointPathRenderer = v284.PlanarGeometryResidualField
    v283.loss_fn = v284.residual_field_loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"v285 device={device} crossfit_points={len(query_ids)}",
        flush=True,
    )

    outer_rows: list[dict[str, object]] = []
    oof_ids: list[np.ndarray] = []
    oof_base: list[np.ndarray] = []
    oof_model: list[np.ndarray] = []
    selected_epochs: list[int] = []
    for outer_fold in range(FOLD_COUNT):
        outer_mask = assigned_fold == outer_fold
        inner_fold = (outer_fold + 1) % FOLD_COUNT
        inner_mask = assigned_fold == inner_fold
        fit_mask = ~(outer_mask | inner_mask)
        fit_rows = np.flatnonzero(fit_mask)
        inner_rows = np.flatnonzero(inner_mask)
        outer_rows_index = np.flatnonzero(outer_mask)

        fit_ids = query_ids[fit_rows]
        fit_feature, full_feature, _test_feature, _ = v264.standardize_features(
            train_raw.take(fit_ids), train_raw, test_raw
        )
        inner_feature = full_feature.take(query_ids[inner_rows])
        _model, inner_training = v283.train_model(
            fit_feature,
            train_s[fit_ids],
            train_h[fit_ids],
            current_sh[fit_rows],
            truth_sh[fit_ids],
            truth_d[fit_ids],
            285 + outer_fold,
            MAX_EPOCHS,
            device,
            len(patches),
            (
                inner_feature,
                train_s[query_ids[inner_rows]],
                train_h[query_ids[inner_rows]],
                current_sh[inner_rows],
                truth_sh[query_ids[inner_rows]],
                truth_d[query_ids[inner_rows]],
            ),
        )
        selected_epoch = int(inner_training["best_epoch"])
        selected_epochs.append(selected_epoch)
        del _model

        train_rows = np.flatnonzero(~outer_mask)
        train_ids = query_ids[train_rows]
        train_feature, full_feature, _test_feature, _ = v264.standardize_features(
            train_raw.take(train_ids), train_raw, test_raw
        )
        outer_feature = full_feature.take(query_ids[outer_rows_index])
        if selected_epoch == 0:
            prediction = current_sh[outer_rows_index].copy()
            gates = np.zeros(len(outer_rows_index), dtype=np.float32)
            outer_training = {
                "seed": 102_850 + outer_fold,
                "best_epoch": 0,
                "history": [],
            }
        else:
            model, outer_training = v283.train_model(
                train_feature,
                train_s[train_ids],
                train_h[train_ids],
                current_sh[train_rows],
                truth_sh[train_ids],
                truth_d[train_ids],
                102_850 + outer_fold,
                selected_epoch,
                device,
                len(patches),
                None,
            )
            prediction, gates = v283.predict(
                model,
                outer_feature,
                train_s[query_ids[outer_rows_index]],
                train_h[query_ids[outer_rows_index]],
                current_sh[outer_rows_index],
                device,
            )
            del model
        baseline = v279.metrics(
            current_sh[outer_rows_index],
            truth_sh[query_ids[outer_rows_index]],
            truth_d[query_ids[outer_rows_index]],
        )
        model_metric = v279.metrics(
            prediction,
            truth_sh[query_ids[outer_rows_index]],
            truth_d[query_ids[outer_rows_index]],
        )
        raw_delta = {
            key: model_metric[key] - baseline[key]
            for key in ("sh", "d", "weighted")
        }
        outer_rows.append(
            {
                "fold": outer_fold,
                "inner_fold": inner_fold,
                "fit_count": int(len(fit_rows)),
                "inner_count": int(len(inner_rows)),
                "outer_count": int(len(outer_rows_index)),
                "selected_epoch": selected_epoch,
                "inner_training": inner_training,
                "outer_training": outer_training,
                "baseline": baseline,
                "model": model_metric,
                "raw_delta": raw_delta,
                "source_mass": summarize(gates),
            }
        )
        oof_ids.append(query_ids[outer_rows_index])
        oof_base.append(current_sh[outer_rows_index])
        oof_model.append(prediction)
        print(
            f"v285 outer={outer_fold} epoch={selected_epoch} "
            f"base={baseline['weighted']:.6f} model={model_metric['weighted']:.6f} "
            f"delta={raw_delta['weighted']:+.6f}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cv_ids = np.concatenate(oof_ids)
    cv_base = np.concatenate(oof_base)
    cv_model = np.concatenate(oof_model)
    alpha, alpha_grid = v279.choose_joint_alpha(
        cv_base, cv_model, truth_sh[cv_ids], truth_d[cv_ids]
    )
    fold_deltas: list[dict[str, float]] = []
    for row, ids, base, model in zip(
        outer_rows, oof_ids, oof_base, oof_model
    ):
        blended = v279.normalize(base + alpha * (model - base))
        metric = v279.metrics(blended, truth_sh[ids], truth_d[ids])
        delta = {
            key: metric[key] - row["baseline"][key]
            for key in ("sh", "d", "weighted")
        }
        row["blended"] = metric
        row["blended_delta"] = delta
        fold_deltas.append(delta)
    mean_delta = {
        key: float(np.mean([value[key] for value in fold_deltas]))
        for key in ("sh", "d", "weighted")
    }
    weighted_wins = int(
        np.sum([value["weighted"] > 0.0 for value in fold_deltas])
    )
    official_selected = bool(
        alpha > 0.0
        and mean_delta["weighted"] >= MIN_OFFICIAL_DELTA
        and mean_delta["sh"] > 0.0
        and mean_delta["d"] > 0.0
        and weighted_wins >= 3
    )

    final_training: list[dict[str, object]] = []
    projection_diagnostics = None
    local_score = None
    if official_selected:
        positive_epochs = [value for value in selected_epochs if value > 0]
        final_epoch = max(int(np.median(positive_epochs)), 1)
        sample_feature, _full_feature, test_feature, _ = v264.standardize_features(
            train_raw.take(query_ids), train_raw, test_raw
        )
        current_test_sh, _current_test_d = v274.extract_current_marginals(BASE)
        seed_predictions: list[np.ndarray] = []
        seed_gates: list[np.ndarray] = []
        for seed in FINAL_SEEDS:
            model, training = v283.train_model(
                sample_feature,
                train_s[query_ids],
                train_h[query_ids],
                current_sh,
                truth_sh[query_ids],
                truth_d[query_ids],
                seed,
                final_epoch,
                device,
                len(patches),
                None,
            )
            prediction, gate = v283.predict(
                model,
                test_feature,
                test_s,
                test_h,
                current_test_sh,
                device,
            )
            seed_predictions.append(prediction)
            seed_gates.append(gate)
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(
                {
                    "version": VERSION,
                    "seed": seed,
                    "epoch": final_epoch,
                    "model_state": model.state_dict(),
                },
                checkpoint,
            )
            final_training.append(
                {**training, "checkpoint": PATHS.relative(checkpoint)}
            )
            del model
        test_model = v279.normalize(np.mean(seed_predictions, axis=0))
        target_sh = v279.normalize(
            current_test_sh + alpha * (test_model - current_test_sh)
        )
        target_d = v279.derive_d_numpy(target_sh)
        np.savez_compressed(
            TARGETS_OUT,
            current_sh=current_test_sh,
            model_sh=test_model,
            target_sh=target_sh,
            target_d=target_d,
            alpha=np.asarray(alpha, dtype=np.float32),
            source_mass=np.mean(seed_gates, axis=0).astype(np.float32),
        )
        projection_diagnostics = v247.project(target_sh, target_d, OUTPUT)
        local_score = v208.score_prediction(OUTPUT)

    report = {
        "method": VERSION,
        "status": (
            "local_gate_passed_needs_packaging"
            if local_score is not None
            and float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "The geometry residual must be trained on a cross-fit current that is isomorphic "
            "to the deployed v239 carrier; ordinary KNN anchors teach the wrong error field."
        ),
        "compliance_boundary": {
            "uses": [
                "official fit-only train H support and PDP",
                "official train/test planar coordinates",
                "official point-cloud full wall-reflection geometry",
                "frozen v239 path decoder cross-fit current",
                "strict v239 official-only test carrier after official selection",
            ],
            "does_not_use": [
                "synthetic/Sionna channel labels",
                "v212/v217 teacher",
                "hidden labels or RT H/path coefficients",
                "local replay truth for training, epoch, or alpha selection",
            ],
            "local_truth_usage": (
                "one final gate score after official selection"
                if official_selected
                else "not read"
            ),
        },
        "sample_reconstruction": sample_report,
        "parameters": {
            "fold_count": FOLD_COUNT,
            "maximum_inner_epochs": MAX_EPOCHS,
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
            "final_seeds": list(FINAL_SEEDS),
            "model": "v284 planar geometry residual field",
        },
        "path_center_spec": center_spec,
        "outer_folds": outer_rows,
        "official_selection": {
            "residual_alpha": alpha,
            "alpha_grid": alpha_grid,
            "mean_delta": mean_delta,
            "weighted_wins": weighted_wins,
            "selected_epochs": selected_epochs,
            "selected": official_selected,
        },
        "final_training": final_training,
        "projection_diagnostics": (
            {
                key: summarize(value)
                for key, value in projection_diagnostics.items()
            }
            if projection_diagnostics is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(
                    float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
                ),
                "gap": float(local_score["score_traceback_leaked"])
                - LOCAL_GATE,
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "samples": PATHS.relative(SAMPLES_OUT),
            "targets": (
                PATHS.relative(TARGETS_OUT) if TARGETS_OUT.is_file() else None
            ),
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
