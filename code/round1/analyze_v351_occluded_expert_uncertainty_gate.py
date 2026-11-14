#!/usr/bin/env python3
"""Medium/heavy joint path expert with epistemic uncertainty rejection."""

from __future__ import annotations

import json

import numpy as np
import torch

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v264_patch_scatterer_transformer as v264
import analyze_v274_planar_barrier_attention as v274
import analyze_v279_joint_component_trajectory as v279
import analyze_v283_joint_geometry_path_renderer as v283
import analyze_v284_planar_geometry_residual_field as v284
import analyze_v285_v239_isomorphic_joint_residual as v285
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 351
VERSION = "v351_occluded_expert_uncertainty_gate"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v351_occluded_expert_oof.npz"
TARGETS_OUT = OUT_DIR / "v351_occluded_expert_targets.npz"
OUTPUT = OUT_DIR / "v351_Round1_Test_Channel.npy"
SAMPLES = PATHS.cache_dir(285) / "v285_v239_crossfit_joint_current.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"

FOLD_COUNT = 5
MAX_EPOCHS = 30
INNER_SEEDS = (351, 1351, 2351)
OUTER_SEED_OFFSET = 103_510
FINAL_SEEDS = (35101, 35131, 35161)
ALPHA_GRID = np.linspace(0.0, 1.25, 11, dtype=np.float32)
KEEP_FRACTIONS = (0.20, 0.35, 0.50, 0.70, 1.00)
MIN_MEAN_DELTA = 0.01
MIN_FOLD_WINS = 4
LOCAL_GATE = 0.645604


def residual_agreement(predictions: list[np.ndarray], base: np.ndarray) -> np.ndarray:
    residuals = [np.asarray(value - base, dtype=np.float64) for value in predictions]
    scores: list[np.ndarray] = []
    for left in range(len(residuals)):
        for right in range(left + 1, len(residuals)):
            a = residuals[left].reshape(len(base), -1)
            b = residuals[right].reshape(len(base), -1)
            scores.append(
                np.sum(a * b, axis=1)
                / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-20)
            )
    return np.mean(np.stack(scores), axis=0).astype(np.float32)


def gate_for_fraction(agreement: np.ndarray, keep_fraction: float) -> np.ndarray:
    if keep_fraction >= 1.0:
        return np.ones(len(agreement), dtype=bool)
    count = max(1, int(np.ceil(keep_fraction * len(agreement))))
    order = np.argsort(agreement)[::-1]
    gate = np.zeros(len(agreement), dtype=bool)
    gate[order[:count]] = True
    return gate


def select_action(
    base: np.ndarray,
    predictions: list[np.ndarray],
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float]], np.ndarray]:
    ensemble = v279.normalize(np.mean(np.stack(predictions), axis=0))
    agreement = residual_agreement(predictions, base)
    baseline = v279.metrics(base, truth_sh, truth_d)
    rows: list[dict[str, float]] = [
        {
            "keep_fraction": 0.0,
            "alpha": 0.0,
            "active_count": 0,
            **baseline,
        }
    ]
    for keep_fraction in KEEP_FRACTIONS:
        gate = gate_for_fraction(agreement, keep_fraction)
        for alpha in ALPHA_GRID[1:]:
            value = base.copy()
            value[gate] = v279.normalize(
                base[gate] + float(alpha) * (ensemble[gate] - base[gate])
            )
            rows.append(
                {
                    "keep_fraction": float(keep_fraction),
                    "alpha": float(alpha),
                    "active_count": int(np.sum(gate)),
                    **v279.metrics(value, truth_sh, truth_d),
                }
            )
    selected = max(rows, key=lambda row: row["weighted"])
    return selected, rows, agreement


def train_seed_ensemble(
    train_feature: v264.FeatureSet,
    train_ids: np.ndarray,
    train_rows: np.ndarray,
    query_feature: v264.FeatureSet,
    query_ids: np.ndarray,
    query_rows: np.ndarray,
    train_s: np.ndarray,
    train_h: np.ndarray,
    current: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
    seeds: tuple[int, ...],
    epochs: list[int] | None,
    device: torch.device,
    patch_count: int,
    select_on_query: bool,
) -> tuple[list[np.ndarray], list[dict[str, object]], list[int]]:
    predictions: list[np.ndarray] = []
    training_rows: list[dict[str, object]] = []
    selected_epochs: list[int] = []
    for seed_index, seed in enumerate(seeds):
        requested_epochs = MAX_EPOCHS if epochs is None else int(epochs[seed_index])
        validation = (
            (
                query_feature,
                train_s[query_ids],
                train_h[query_ids],
                current[query_rows],
                truth_sh[query_ids],
                truth_d[query_ids],
            )
            if select_on_query
            else None
        )
        model, training = v283.train_model(
            train_feature,
            train_s[train_ids],
            train_h[train_ids],
            current[train_rows],
            truth_sh[train_ids],
            truth_d[train_ids],
            seed,
            requested_epochs,
            device,
            patch_count,
            validation,
        )
        selected_epoch = int(training["best_epoch"] if select_on_query else requested_epochs)
        selected_epochs.append(selected_epoch)
        prediction, gates = v283.predict(
            model,
            query_feature,
            train_s[query_ids],
            train_h[query_ids],
            current[query_rows],
            device,
        )
        predictions.append(prediction)
        training_rows.append(
            {
                "seed": int(seed),
                "selected_epoch": selected_epoch,
                "training": training,
                "source_mass": summarize(gates),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predictions, training_rows, selected_epochs


def main() -> None:
    for path in (
        SAMPLES,
        TARGET_CACHE,
        BASE,
        v283.TRAIN_CANDIDATES,
        v283.TEST_CANDIDATES,
        v283.SCENE_CACHE,
        v283.V265_REPORT,
        v285.v281.v238.GEOM_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    v283.JointPathRenderer = v284.PlanarGeometryResidualField
    v283.loss_fn = v284.residual_field_loss
    sample = np.load(SAMPLES)
    query_ids = sample["query_ids"].astype(np.int64)
    assigned_fold = sample["assigned_fold"].astype(np.int16)
    current = sample["current_sh"].astype(np.float32)
    target_data = np.load(TARGET_CACHE)
    truth_sh = v279.normalize(target_data["sh"].astype(np.float32))
    truth_d = v279.normalize(target_data["d"].astype(np.float32))

    geometry_cache = np.load(v285.v281.v238.GEOM_CACHE)
    train_regime = regime_of(geometry_cache["train_blocked"].astype(np.float32))
    test_regime = regime_of(geometry_cache["test_blocked"].astype(np.float32))
    query_expert = np.isin(train_regime[query_ids].astype(str), ["medium", "heavy"])

    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    patches = np.load(v283.SCENE_CACHE)["wall_patches"].astype(np.float32)
    train_candidates = dict(np.load(v283.TRAIN_CANDIDATES))
    test_candidates = dict(np.load(v283.TEST_CANDIDATES))
    train_node, test_node = v257.build_node_features(train_pos[:, :2], test_pos[:, :2])
    train_raw = v264.build_features(train_pos, train_candidates, patches, train_node)
    test_raw = v264.build_features(test_pos, test_candidates, patches, test_node)
    with v283.V265_REPORT.open("r", encoding="utf-8") as handle:
        center_spec = json.load(handle)["selection"]["best"]
    train_s, train_h = v283.centers(train_pos, train_candidates, train_raw, center_spec)
    test_s, test_h = v283.centers(test_pos, test_candidates, test_raw, center_spec)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"v351 device={device} samples={len(query_ids)} expert={int(np.sum(query_expert))}",
        flush=True,
    )

    fold_rows: list[dict[str, object]] = []
    oof_ids: list[np.ndarray] = []
    oof_base: list[np.ndarray] = []
    oof_prediction: list[np.ndarray] = []
    selected_epoch_rows: list[list[int]] = []
    selected_actions: list[dict[str, float]] = []
    for fold in range(FOLD_COUNT):
        outer_rows_all = np.flatnonzero(assigned_fold == fold)
        outer_rows = outer_rows_all[query_expert[outer_rows_all]]
        inner_fold = (fold + 1) % FOLD_COUNT
        inner_rows_all = np.flatnonzero(assigned_fold == inner_fold)
        inner_rows = inner_rows_all[query_expert[inner_rows_all]]
        fit_rows = np.flatnonzero(
            (assigned_fold != fold)
            & (assigned_fold != inner_fold)
            & query_expert
        )

        fit_ids = query_ids[fit_rows]
        fit_feature, full_feature, _test_feature, _ = v264.standardize_features(
            train_raw.take(fit_ids), train_raw, test_raw
        )
        inner_feature = full_feature.take(query_ids[inner_rows])
        inner_predictions, inner_training, epochs = train_seed_ensemble(
            fit_feature,
            fit_ids,
            fit_rows,
            inner_feature,
            query_ids[inner_rows],
            inner_rows,
            train_s,
            train_h,
            current,
            truth_sh,
            truth_d,
            tuple(seed + fold for seed in INNER_SEEDS),
            None,
            device,
            len(patches),
            True,
        )
        inner_selected, inner_grid, inner_agreement = select_action(
            current[inner_rows],
            inner_predictions,
            truth_sh[query_ids[inner_rows]],
            truth_d[query_ids[inner_rows]],
        )
        selected_epoch_rows.append(epochs)
        selected_actions.append(inner_selected)

        train_rows = np.flatnonzero((assigned_fold != fold) & query_expert)
        train_ids = query_ids[train_rows]
        train_feature, full_feature, _test_feature, _ = v264.standardize_features(
            train_raw.take(train_ids), train_raw, test_raw
        )
        outer_feature = full_feature.take(query_ids[outer_rows])
        outer_predictions, outer_training, _ = train_seed_ensemble(
            train_feature,
            train_ids,
            train_rows,
            outer_feature,
            query_ids[outer_rows],
            outer_rows,
            train_s,
            train_h,
            current,
            truth_sh,
            truth_d,
            tuple(OUTER_SEED_OFFSET + 100 * fold + i for i in range(3)),
            epochs,
            device,
            len(patches),
            False,
        )
        ensemble = v279.normalize(np.mean(np.stack(outer_predictions), axis=0))
        outer_agreement = residual_agreement(outer_predictions, current[outer_rows])
        gate = gate_for_fraction(
            outer_agreement, float(inner_selected["keep_fraction"])
        ) if float(inner_selected["keep_fraction"]) > 0.0 else np.zeros(len(outer_rows), dtype=bool)
        prediction = current[outer_rows].copy()
        if np.any(gate):
            alpha = float(inner_selected["alpha"])
            prediction[gate] = v279.normalize(
                current[outer_rows][gate]
                + alpha * (ensemble[gate] - current[outer_rows][gate])
            )
        base_metric = v279.metrics(
            current[outer_rows], truth_sh[query_ids[outer_rows]], truth_d[query_ids[outer_rows]]
        )
        metric = v279.metrics(
            prediction, truth_sh[query_ids[outer_rows]], truth_d[query_ids[outer_rows]]
        )
        delta = {key: float(metric[key] - base_metric[key]) for key in ("sh", "d", "weighted")}
        fold_rows.append(
            {
                "fold": fold,
                "inner_fold": inner_fold,
                "fit_count": int(len(fit_rows)),
                "inner_count": int(len(inner_rows)),
                "outer_count": int(len(outer_rows)),
                "inner_training": inner_training,
                "inner_action_grid": inner_grid,
                "selected_action": inner_selected,
                "selected_epochs": epochs,
                "outer_training": outer_training,
                "inner_agreement": summarize(inner_agreement),
                "outer_agreement": summarize(outer_agreement),
                "active_count": int(np.sum(gate)),
                "baseline": base_metric,
                "prediction": metric,
                "delta": delta,
            }
        )
        oof_ids.append(query_ids[outer_rows])
        oof_base.append(current[outer_rows])
        oof_prediction.append(prediction)
        print(
            f"v351 fold={fold} epochs={epochs} keep={inner_selected['keep_fraction']:.2f} "
            f"alpha={inner_selected['alpha']:.3f} delta={delta['weighted']:+.6f}",
            flush=True,
        )

    mean_delta = {
        key: float(np.mean([row["delta"][key] for row in fold_rows]))
        for key in ("sh", "d", "weighted")
    }
    wins = int(sum(row["delta"]["weighted"] > 0.0 for row in fold_rows))
    official_selected = bool(
        mean_delta["weighted"] >= MIN_MEAN_DELTA and wins >= MIN_FOLD_WINS
    )
    np.savez_compressed(
        OOF_OUT,
        ids=np.concatenate(oof_ids),
        base=np.concatenate(oof_base),
        prediction=np.concatenate(oof_prediction),
    )

    final_training: list[dict[str, object]] = []
    local_score = None
    projection = None
    if official_selected:
        final_epochs = [
            max(1, int(np.median([row[index] for row in selected_epoch_rows])))
            for index in range(3)
        ]
        positive_actions = [row for row in selected_actions if row["keep_fraction"] > 0.0]
        final_keep = float(np.median([row["keep_fraction"] for row in positive_actions]))
        final_alpha = float(np.median([row["alpha"] for row in positive_actions]))
        train_rows = np.flatnonzero(query_expert)
        train_ids = query_ids[train_rows]
        train_feature, _full_feature, final_test_feature, _ = v264.standardize_features(
            train_raw.take(train_ids), train_raw, test_raw
        )
        current_test, _ = v274.extract_current_marginals(BASE)
        test_expert_ids = np.flatnonzero(
            np.isin(test_regime.astype(str), ["medium", "heavy"])
        ).astype(np.int64)
        test_expert_feature = final_test_feature.take(test_expert_ids)
        test_predictions: list[np.ndarray] = []
        for seed_index, seed in enumerate(FINAL_SEEDS):
            model, training = v283.train_model(
                train_feature,
                train_s[train_ids],
                train_h[train_ids],
                current[train_rows],
                truth_sh[train_ids],
                truth_d[train_ids],
                seed,
                final_epochs[seed_index],
                device,
                len(patches),
                None,
            )
            value, source_mass = v283.predict(
                model,
                test_expert_feature,
                test_s[test_expert_ids],
                test_h[test_expert_ids],
                current_test[test_expert_ids],
                device,
            )
            test_predictions.append(value)
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(
                {
                    "version": VERSION,
                    "seed": seed,
                    "epoch": final_epochs[seed_index],
                    "model_state": model.state_dict(),
                },
                checkpoint,
            )
            final_training.append(
                {
                    "seed": seed,
                    "epoch": final_epochs[seed_index],
                    "training": training,
                    "source_mass": summarize(source_mass),
                    "checkpoint": PATHS.relative(checkpoint),
                }
            )
        ensemble = v279.normalize(np.mean(np.stack(test_predictions), axis=0))
        agreement = residual_agreement(
            test_predictions, current_test[test_expert_ids]
        )
        gate = gate_for_fraction(agreement, final_keep)
        target_sh = current_test.copy()
        target_sh[test_expert_ids[gate]] = v279.normalize(
            current_test[test_expert_ids[gate]]
            + final_alpha
            * (ensemble[gate] - current_test[test_expert_ids[gate]])
        )
        target_d = v279.derive_d_numpy(target_sh)
        np.savez_compressed(
            TARGETS_OUT,
            current_sh=current_test,
            target_sh=target_sh,
            target_d=target_d,
            expert_ids=test_expert_ids,
            agreement=agreement,
            gate=gate,
            alpha=np.asarray(final_alpha, dtype=np.float32),
            keep_fraction=np.asarray(final_keep, dtype=np.float32),
        )
        projection = v247.project(target_sh, target_d, OUTPUT)
        local_score = v208.score_prediction(OUTPUT)

    report = {
        "method": VERSION,
        "status": "invalidated_crossfold_current_leakage",
        "round": 4,
        "validity_audit": {
            "strict_oof_usable": False,
            "invalidated_by": "v353_crossfold_current_leakage_audit",
            "reason": "reused v285 currents contain labels from the new outer fold",
            "numerical_selection_before_audit": official_selected,
        },
        "primary_hypothesis": (
            "Joint path birth/death and delay-angle correction is a distinct medium/heavy "
            "problem; an occlusion specialist should act only where independent seeds agree "
            "on the correction direction."
        ),
        "compliance": {
            "uses": [
                "official train H labels",
                "official train/test coordinates",
                "official point-cloud path and blocker features",
                "strict fit-only v239-like cross-fit current",
            ],
            "does_not_use": [
                "v207/v212/synthetic targets or teacher outputs",
                "restored scene/material/RT coefficients",
                "online labels for model or gate selection",
            ],
            "local_truth_usage": (
                "one frozen diagnostic after official OOF selection"
                if official_selected
                else "not read"
            ),
        },
        "parameters": {
            "expert_regimes": ["medium", "heavy"],
            "ensemble_size": 3,
            "maximum_inner_epochs": MAX_EPOCHS,
            "alpha_grid": ALPHA_GRID.tolist(),
            "keep_fractions": list(KEEP_FRACTIONS),
            "minimum_mean_delta": MIN_MEAN_DELTA,
            "minimum_fold_wins": MIN_FOLD_WINS,
        },
        "fold_rows": fold_rows,
        "official_selection": {
            "mean_delta": mean_delta,
            "fold_wins": wins,
            "selected": False,
            "selected_numerically_before_audit": official_selected,
        },
        "final_training": final_training,
        "projection": (
            {key: summarize(value) for key, value in projection.items()}
            if projection is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(local_score["score_traceback_leaked"] >= LOCAL_GATE),
                "gap": float(local_score["score_traceback_leaked"] - LOCAL_GATE),
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "oof": PATHS.relative(OOF_OUT),
            "targets": PATHS.relative(TARGETS_OUT) if TARGETS_OUT.is_file() else None,
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["official_selection"], ensure_ascii=False, indent=2), flush=True)
    if local_score is not None:
        print(json.dumps(local_score, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
