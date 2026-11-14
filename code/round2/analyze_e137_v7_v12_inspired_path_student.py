#!/usr/bin/env python3
"""Test whether V12-revealed path corrections are learnable from V7 and official data."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import json
import math
import time

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e119_v12_unique_path_complement as e119
import analyze_e123_complete_path_variable_system as e123
import analyze_e131_v7_path_selection_evidence as e131
import analyze_e133_v7_deformable_peak_alignment as e133
import analyze_e134_sionna_candidate_selection_evidence as e134
import analyze_e135_sionna_selection_v12_complete_score as e135
import train_e115_complete_path_state_operator as e115
import train_e118_hierarchical_unique_path_operator as e118
import train_e130_official_path_identity_system as e130
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E137-v7-v12-inspired-path-student"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v7_v12_inspired_path_student"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v7_v12_inspired_path_student"
REPORT = ANALYSIS_DIR / "v7_v12_inspired_path_student.json"
METRICS = ARTIFACT_DIR / "v7_v12_inspired_path_student_metrics.npz"
V7_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "full_sionna_migration"
    / "oof_v7_nested_channel.npy"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

TOP_K = e131.TOP_K
PATCH_OFFSETS = tuple(
    (horizontal, vertical, delay)
    for horizontal in range(-1, 2)
    for vertical in range(-1, 2)
    for delay in range(-2, 3)
)
MODEL_CONFIG = {
    "n_estimators": 256,
    "max_depth": 18,
    "min_samples_leaf": 12,
    "max_features": 0.65,
    "bootstrap": False,
    "n_jobs": -1,
}
ACTION = e119.Action(
    endpoint="predicted_endpoint",
    mode="sqrt",
    pas_beta=0.10,
    pdp_beta=0.10,
    order="pas_pdp",
    cycles=4,
)
MINIMUM_RECALL_GAIN = 0.010
MINIMUM_COMPLETE_GAIN = 0.002
MINIMUM_POSITIVE_FOLDS = 4
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--trees", type=int, default=MODEL_CONFIG["n_estimators"])
    parser.add_argument("--seed", type=int, default=20_260_809_137)
    return parser.parse_args()


def normalized_log(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    scale = np.max(np.where(mask, positive, 0.0), axis=1, keepdims=True)
    relative = np.divide(
        positive,
        np.maximum(scale, TINY),
        out=np.zeros_like(positive),
    )
    return np.where(mask, np.log10(np.maximum(relative, 1e-10)) / 10.0 + 1.0, 0.0).astype(
        np.float32
    )


def normalized_target(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(value, dtype=np.float64), 0.0) * mask
    total = positive.sum(axis=1, keepdims=True)
    normalized = np.divide(
        positive,
        np.maximum(total, TINY),
        out=np.zeros_like(positive),
    )
    return np.sqrt(normalized).astype(np.float32)


def gather_outer_array(
    query_ids: np.ndarray,
    query_fold: np.ndarray,
    path_for_outer,
    name: str | None = None,
) -> np.ndarray:
    output = None
    for outer in range(5):
        rows = np.flatnonzero(query_fold == outer)
        source_path = path_for_outer(outer)
        if name is not None:
            source_path = source_path / f"{name}.npy"
        source = np.load(source_path, mmap_mode="r")
        if output is None:
            output = np.empty((len(query_ids), *source.shape[1:]), dtype=source.dtype)
        output[rows] = source[query_ids[rows]]
    if output is None:
        raise RuntimeError("no outer-fold arrays were gathered")
    return output


def v7_patch_features(
    support: np.ndarray,
    coordinate: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    count, candidates, _ = coordinate.shape
    center = np.rint(coordinate).astype(np.int64)
    patch = np.zeros((count, candidates, len(PATCH_OFFSETS)), dtype=np.float32)
    rows = np.arange(count, dtype=np.int64)[:, None]
    for index, (horizontal, vertical, delay) in enumerate(PATCH_OFFSETS):
        h_index = (center[..., 0] + horizontal) % e115.MH
        v_index = (center[..., 1] + vertical) % e115.MV
        d_index = (center[..., 2] + delay) % e115.S_SC
        patch[..., index] = np.asarray(
            support[rows, h_index, v_index, d_index], dtype=np.float32
        )
    patch *= mask[..., None]
    point_scale = np.max(patch, axis=(1, 2), keepdims=True)
    relative = np.divide(
        patch,
        np.maximum(point_scale, TINY),
        out=np.zeros_like(patch),
    )
    log_relative = np.log10(np.maximum(relative, 1e-10)) / 10.0 + 1.0
    local_total = patch.sum(axis=2, keepdims=True, dtype=np.float64)
    local_shape = np.divide(
        patch,
        np.maximum(local_total, TINY),
        out=np.zeros_like(patch, dtype=np.float64),
    )
    return np.concatenate(
        (log_relative.astype(np.float32), np.sqrt(local_shape).astype(np.float32)),
        axis=2,
    )


def query_geometry_features(
    query_ids: np.ndarray,
    site: np.ndarray,
) -> np.ndarray:
    setup, positions, _, _, _ = load_positions()
    point = positions[query_ids].astype(np.float32)
    bs = setup.bs_positions[site].astype(np.float32)
    delta = point - bs
    distance = np.linalg.norm(delta, axis=1, keepdims=True)
    azimuth = np.arctan2(delta[:, 1], delta[:, 0])[:, None]
    return np.concatenate(
        (
            point[:, :2] / 200.0,
            delta[:, :2] / 200.0,
            distance / 200.0,
            np.sin(azimuth),
            np.cos(azimuth),
            (site == 0)[:, None].astype(np.float32),
            (site == 1)[:, None].astype(np.float32),
        ),
        axis=1,
    ).astype(np.float32)


def build_features(
    query_ids: np.ndarray,
    query_fold: np.ndarray,
    site: np.ndarray,
    mask: np.ndarray,
    coordinate: np.ndarray,
    evidence: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    base_rank = e131.row_ranks(evidence["base_mass"], mask)
    v7_rank = e131.row_ranks(evidence["v7_local_energy"], mask)
    neighbor_rank = e131.row_ranks(evidence["v7_neighbor_energy"], mask)
    continuity_rank = e131.row_ranks(evidence["v7_continuity_energy"], mask)
    scalar = np.stack(
        (
            base_rank,
            v7_rank,
            neighbor_rank,
            continuity_rank,
            normalized_log(evidence["base_mass"], mask),
            normalized_log(evidence["v7_local_energy"], mask),
            normalized_log(evidence["v7_neighbor_energy"], mask),
            normalized_log(evidence["v7_continuity_energy"], mask),
        ),
        axis=2,
    )
    angle = 2.0 * math.pi * coordinate / e115.PERIODS[None, None]
    periodic_coordinate = np.concatenate((np.sin(angle), np.cos(angle)), axis=2).astype(
        np.float32
    )
    support = np.load(e131.V7_SUPPORT, mmap_mode="r")
    patch = v7_patch_features(support, coordinate, mask)
    query = query_geometry_features(query_ids, site)
    query = np.broadcast_to(query[:, None], (len(query), mask.shape[1], query.shape[1]))

    hierarchy = lambda outer: e118.cache_dir(e123.cache_tag(outer))
    cluster = gather_outer_array(query_ids, query_fold, hierarchy, "cluster_feature")
    member = gather_outer_array(query_ids, query_fold, hierarchy, "member_count")
    log_port = gather_outer_array(query_ids, query_fold, hierarchy, "base_log_port")
    phase = gather_outer_array(query_ids, query_fold, hierarchy, "base_phase")
    trajectory = gather_outer_array(
        query_ids, query_fold, e130.trajectory_cache_path
    )
    map_feature = gather_outer_array(
        query_ids, query_fold, e130.map_feature_cache_path
    )
    port = np.exp(np.clip(log_port.astype(np.float32), -20.0, 10.0))
    port /= np.maximum(port.sum(axis=2, keepdims=True), TINY)
    basic_parts = (
        scalar,
        periodic_coordinate,
        patch,
        query.astype(np.float32),
        port.astype(np.float32),
        phase.astype(np.float32),
    )
    basic = np.concatenate(basic_parts, axis=2)
    full = np.concatenate(
        (
            basic,
            cluster.astype(np.float32),
            np.log1p(member.astype(np.float32))[..., None],
            trajectory.astype(np.float32),
            map_feature.astype(np.float32),
        ),
        axis=2,
    )
    basic = np.nan_to_num(basic, nan=0.0, posinf=8.0, neginf=-8.0)
    full = np.nan_to_num(full, nan=0.0, posinf=8.0, neginf=-8.0)
    basic *= mask[..., None]
    full *= mask[..., None]
    return basic, full, {
        "scalar": int(scalar.shape[2]),
        "periodic_coordinate": int(periodic_coordinate.shape[2]),
        "v7_patch": int(patch.shape[2]),
        "query_geometry": int(query.shape[2]),
        "base_port": int(port.shape[2]),
        "base_phase": int(phase.shape[2]),
        "cluster": int(cluster.shape[2]),
        "member_count": 1,
        "official_trajectory": int(trajectory.shape[2]),
        "candidate_map": int(map_feature.shape[2]),
        "basic_total": int(basic.shape[2]),
        "full_total": int(full.shape[2]),
    }


def crossfit_regressor(
    features: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    fold: np.ndarray,
    args: argparse.Namespace,
    name: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    output = np.zeros(mask.shape, dtype=np.float32)
    records = []
    for outer in range(5):
        fit_rows = np.flatnonzero(fold != outer)
        held_rows = np.flatnonzero(fold == outer)
        fit_valid = mask[fit_rows]
        held_valid = mask[held_rows]
        x_fit = features[fit_rows][fit_valid]
        y_fit = target[fit_rows][fit_valid]
        sample_weight = 1.0 + 5.0 * y_fit
        model = ExtraTreesRegressor(
            **{**MODEL_CONFIG, "n_estimators": args.trees},
            random_state=(args.seed + 1009 * outer + 7919 * len(name)) % (2**32),
        )
        model.fit(x_fit, y_fit, sample_weight=sample_weight)
        prediction = model.predict(features[held_rows][held_valid]).astype(np.float32)
        local = np.zeros(mask[held_rows].shape, dtype=np.float32)
        local[held_valid] = prediction
        output[held_rows] = local
        records.append(
            {
                "outer": outer,
                "fit_points": int(len(fit_rows)),
                "held_points": int(len(held_rows)),
                "fit_candidates": int(fit_valid.sum()),
                "held_candidates": int(held_valid.sum()),
                "tree_count": int(args.trees),
            }
        )
        del model, x_fit, y_fit, sample_weight, prediction, local
        gc.collect()
        print(f"{name} outer={outer} complete", flush=True)
    output *= mask
    return output, records


def selection_from_prediction(
    prediction: np.ndarray,
    base_rank: np.ndarray,
    v7_rank: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_rank = e131.row_ranks(prediction, mask)
    score = (base_rank + v7_rank + predicted_rank) / 3.0
    return score, e131.top_k_mask(score, mask)


def fold_gains(
    candidate: dict[str, object],
    baseline: dict[str, object],
) -> list[dict[str, float | int]]:
    return [
        {
            "fold": int(left["fold"]),
            "gain": float(left["score"]) - float(right["score"]),
        }
        for left, right in zip(candidate["folds"], baseline["folds"], strict=True)
    ]


def evaluate_complete_v7(
    selections: OrderedDict[str, np.ndarray],
    query_ids: np.ndarray,
    fold: np.ndarray,
    coordinate: np.ndarray,
    base_mass: np.ndarray,
    base_port: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, object], dict[str, tuple[np.ndarray, ...]]]:
    base_np = np.load(V7_CHANNEL, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    truth_np = np.asarray(truth_all[query_ids], dtype=np.complex64)
    base = torch.as_tensor(np.array(base_np, dtype=np.complex64, copy=True), device=device)
    truth = torch.as_tensor(truth_np, device=device)
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    baseline_point = point_metrics(base, truth, truth_pas, truth_pdp)
    baseline = statistics(*baseline_point)
    if abs(float(baseline["score"]) - 0.6444563457343335) > 2e-6:
        raise RuntimeError(f"unexpected V7 canonical score: {baseline['score']}")

    rows = OrderedDict()
    point_cache: dict[str, tuple[np.ndarray, ...]] = {"v7_identity": baseline_point}
    for name, selection in selections.items():
        mass = e133.selected_mass(base_mass, selection, mask)
        endpoint_pas, endpoint_pdp = e135.render_candidate_endpoint(
            coordinate, mass, base_port, device, batch_size
        )
        action = e119.Action(
            endpoint=name,
            mode=ACTION.mode,
            pas_beta=ACTION.pas_beta,
            pdp_beta=ACTION.pdp_beta,
            order=ACTION.order,
            cycles=ACTION.cycles,
        )
        with torch.inference_mode():
            prediction = e119.make_candidate(
                base,
                base_pas,
                base_pdp,
                *e119.expand_endpoint(endpoint_pas, endpoint_pdp, device),
                action,
            )
            values = point_metrics(prediction, truth, truth_pas, truth_pdp)
        metrics = statistics(*values)
        folds = []
        for outer in range(5):
            selected = fold == outer
            local = statistics(*values, selected)
            anchor = statistics(*baseline_point, selected)
            folds.append(
                {
                    "fold": outer,
                    "count": int(selected.sum()),
                    **local,
                    "delta_to_v7": float(local["score"] - anchor["score"]),
                }
            )
        rows[name] = {
            "action": {
                "mode": action.mode,
                "pas_beta": action.pas_beta,
                "pdp_beta": action.pdp_beta,
                "order": action.order,
                "cycles": action.cycles,
            },
            "metrics": metrics,
            "delta_to_v7": float(metrics["score"] - baseline["score"]),
            "positive_folds": int(sum(row["delta_to_v7"] > 0.0 for row in folds)),
            "worst_fold_delta": float(min(row["delta_to_v7"] for row in folds)),
            "folds": folds,
        }
        point_cache[name] = values
        print(
            f"complete V7 {name}: score={metrics['score']:.9f} "
            f"delta={metrics['score'] - baseline['score']:+.9f}",
            flush=True,
        )
    return {"baseline_v7": baseline, "actions": rows}, point_cache


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    v7 = np.load(e131.METRICS, allow_pickle=False)
    teacher = np.load(e134.METRICS, allow_pickle=False)
    query_ids = v7["query_ids"].astype(np.int64)
    fold = v7["candidate_fold"].astype(np.int8)
    site = v7["query_site"].astype(np.int8)
    mask = v7["candidate_mask"].astype(bool)
    if not np.array_equal(query_ids, teacher["query_ids"].astype(np.int64)):
        raise RuntimeError("E131/E134 query mismatch")
    candidate = e131.gather_candidate_cache(query_ids, fold)
    coordinate = candidate["coordinate"].astype(np.float32)
    basic_features, full_features, feature_groups = build_features(
        query_ids, fold, site, mask, coordinate, v7
    )
    official_target = normalized_target(v7["target_mass"], mask)
    sionna_target = normalized_target(
        teacher["evidence_corrected_full_one_to_one"], mask
    )

    official_basic, official_basic_records = crossfit_regressor(
        basic_features, official_target, mask, fold, args, "official_basic"
    )
    official_full, official_full_records = crossfit_regressor(
        full_features, official_target, mask, fold, args, "official_full"
    )
    sionna_full, sionna_full_records = crossfit_regressor(
        full_features, sionna_target, mask, fold, args, "sionna_full"
    )

    base_rank = e131.row_ranks(v7["base_mass"], mask)
    v7_rank = e131.row_ranks(v7["v7_local_energy"], mask)
    score_and_selection: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
    score_and_selection["base_plus_v7"] = (
        0.5 * (base_rank + v7_rank),
        teacher["selected_base_plus_v7"].astype(bool),
    )
    direct_sionna_rank = e131.row_ranks(
        teacher["evidence_corrected_full_one_to_one"], mask
    )
    score_and_selection["direct_sionna_teacher"] = (
        (base_rank + v7_rank + direct_sionna_rank) / 3.0,
        teacher["selected_base_v7_plus_corrected_full_one_to_one"].astype(bool),
    )
    for name, prediction in (
        ("official_basic_student", official_basic),
        ("official_full_student", official_full),
        ("sionna_distilled_student", sionna_full),
    ):
        score_and_selection[name] = selection_from_prediction(
            prediction, base_rank, v7_rank, mask
        )

    ranking = OrderedDict()
    recall_cache = {}
    for name, (score, selection) in score_and_selection.items():
        ranking[name], recall_cache[name] = e131.ranking_metrics(
            score,
            selection,
            v7["target_mass"].astype(np.float32),
            mask,
            fold,
            site,
        )
        print(
            f"ranking {name}: recall={ranking[name]['true_mass_recall']:.6f}",
            flush=True,
        )

    component = np.load(e114.COMPONENTS, allow_pickle=False)
    actual_rollout = e134.evaluate_actual_states(
        OrderedDict((name, value[1]) for name, value in score_and_selection.items()),
        query_ids,
        fold,
        mask,
        coordinate,
        component["coordinates"][query_ids].astype(np.float32),
        component["coefficients"][query_ids].astype(np.complex64),
        component["point_power"][query_ids].astype(np.float64),
        device,
        args.batch_size,
    )

    full_cache = e123.collect_cache_arrays(query_ids, fold)
    base_mass = e123.normalize_mass(
        np.exp(np.clip(full_cache["base_log_mass"].astype(np.float32), -30.0, 20.0)),
        mask,
    )
    base_port = e123.normalize_port(
        np.exp(np.clip(full_cache["base_log_port"].astype(np.float32), -20.0, 10.0))
    ).reshape(len(query_ids), -1, e115.MP, e115.N_UE)
    complete, complete_points = evaluate_complete_v7(
        OrderedDict((name, value[1]) for name, value in score_and_selection.items()),
        query_ids,
        fold,
        coordinate,
        base_mass,
        base_port,
        mask,
        device,
        args.batch_size,
    )

    baseline_recall = float(ranking["base_plus_v7"]["true_mass_recall"])
    primary_recall = float(ranking["official_full_student"]["true_mass_recall"])
    recall_fold_gain = [
        float(left["value"] - right["value"])
        for left, right in zip(
            ranking["official_full_student"]["fold_true_mass_recall"],
            ranking["base_plus_v7"]["fold_true_mass_recall"],
            strict=True,
        )
    ]
    primary_complete = complete["actions"]["official_full_student"]
    direct_teacher_recall_gain = float(
        ranking["direct_sionna_teacher"]["true_mass_recall"] - baseline_recall
    )
    basic_recall_gain = float(
        ranking["official_basic_student"]["true_mass_recall"] - baseline_recall
    )
    recall_information_passed = bool(
        primary_recall - baseline_recall >= MINIMUM_RECALL_GAIN
        and sum(value > 0.0 for value in recall_fold_gain) >= MINIMUM_POSITIVE_FOLDS
    )
    gate = {
        "primary": "official_full_student",
        "minimum_true_mass_recall_gain": MINIMUM_RECALL_GAIN,
        "minimum_complete_v7_score_gain": MINIMUM_COMPLETE_GAIN,
        "minimum_positive_folds": MINIMUM_POSITIVE_FOLDS,
        "true_mass_recall_gain": float(primary_recall - baseline_recall),
        "true_mass_recall_fold_gains": recall_fold_gain,
        "true_mass_recall_positive_folds": int(sum(value > 0.0 for value in recall_fold_gain)),
        "complete_v7_score_gain": float(primary_complete["delta_to_v7"]),
        "complete_v7_positive_folds": int(primary_complete["positive_folds"]),
        "complete_v7_worst_fold_delta": float(primary_complete["worst_fold_delta"]),
    }
    gate["passed"] = bool(
        recall_information_passed
        and gate["complete_v7_score_gain"] >= MINIMUM_COMPLETE_GAIN
        and gate["complete_v7_positive_folds"] >= MINIMUM_POSITIVE_FOLDS
    )

    attribution = {
        "path_selection_information_partially_learnable": recall_information_passed,
        "official_full_fraction_of_direct_teacher_recall_gain": float(
            (primary_recall - baseline_recall) / max(direct_teacher_recall_gain, TINY)
        ),
        "official_basic_true_mass_recall_gain": basic_recall_gain,
        "extra_geometry_recall_delta_full_minus_basic": float(
            ranking["official_full_student"]["true_mass_recall"]
            - ranking["official_basic_student"]["true_mass_recall"]
        ),
        "direct_teacher_complete_v7_gain": float(
            complete["actions"]["direct_sionna_teacher"]["delta_to_v7"]
        ),
        "official_full_increment_over_unlearned_base_endpoint": float(
            complete["actions"]["official_full_student"]["metrics"]["score"]
            - complete["actions"]["base_plus_v7"]["metrics"]["score"]
        ),
        "official_full_complete_delta_to_direct_teacher": float(
            complete["actions"]["official_full_student"]["metrics"]["score"]
            - complete["actions"]["direct_sionna_teacher"]["metrics"]["score"]
        ),
        "current_interface_is_primary_limit": bool(
            complete["actions"]["direct_sionna_teacher"]["delta_to_v7"]
            < MINIMUM_COMPLETE_GAIN
        ),
    }

    payload = {
        "experiment": VERSION,
        "status": "small_learnability_screen_complete",
        "primary_hypothesis": (
            "The path birth/death and energy corrections revealed by V12/Sionna can "
            "be predicted at unseen holes from the strict V7 field plus official-H "
            "trajectory and official-map candidate features."
        ),
        "protocol": {
            "canonical_fixed500": True,
            "crossfit_by_candidate_spatial_fold": True,
            "held_h_role": "ranking labels and final scoring only",
            "v7_role": "official-only input carrier and local angle-delay evidence",
            "sionna_role": (
                "direct diagnostic teacher and training labels for the secondary "
                "distillation control; never a held-fold input"
            ),
            "primary_student_training_target": "official held-path mass on training folds",
            "primary_student_inference_inputs": (
                "V7 local field, official-H donor trajectories, official map, coordinates"
            ),
            "top_k": TOP_K,
            "model": {**MODEL_CONFIG, "n_estimators": args.trees},
            "projection_action": {
                "mode": ACTION.mode,
                "pas_beta": ACTION.pas_beta,
                "pdp_beta": ACTION.pdp_beta,
                "order": ACTION.order,
                "cycles": ACTION.cycles,
            },
            "parameter_scan": False,
        },
        "feature_groups": feature_groups,
        "training": {
            "official_basic": official_basic_records,
            "official_full": official_full_records,
            "sionna_full": sionna_full_records,
        },
        "ranking": ranking,
        "actual_predicted_state_rollout": actual_rollout,
        "complete_v7_rollout": complete,
        "attribution": attribution,
        "gate": gate,
        "decision": {
            "path_selection_information_partially_learnable": recall_information_passed,
            "current_top15_projection_sufficient_for_scale_up": bool(gate["passed"]),
            "large_model_authorized": bool(gate["passed"]),
            "candidate_or_zip_allowed": False,
            "next_action": (
                "scale_official_v7_path_student"
                if gate["passed"]
                else "replace_top15_selection_with_conditional_path_state_correction"
            ),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        candidate_fold=fold,
        candidate_mask=mask,
        official_basic_prediction=official_basic,
        official_full_prediction=official_full,
        sionna_full_prediction=sionna_full,
        **{
            f"selected_{name}": value[1]
            for name, value in score_and_selection.items()
        },
        **{
            f"point_recall_{name}": value.astype(np.float32)
            for name, value in recall_cache.items()
        },
        **{
            f"complete_{name}_{kind}": np.asarray(values[index], dtype=np.float64)
            for name, values in complete_points.items()
            for index, kind in enumerate(("pas", "pdp", "squared_error", "truth_energy"))
        },
    )
    print(
        json.dumps(
            {
                "ranking": {
                    name: value["true_mass_recall"] for name, value in ranking.items()
                },
                "actual_scores": {
                    name: value["all"]["score"]
                    for name, value in actual_rollout.items()
                },
                "complete_v7": {
                    "baseline": complete["baseline_v7"],
                    "actions": {
                        name: {
                            "score": value["metrics"]["score"],
                            "delta": value["delta_to_v7"],
                            "positive_folds": value["positive_folds"],
                        }
                        for name, value in complete["actions"].items()
                    },
                },
                "gate": gate,
                "decision": payload["decision"],
                "seconds": payload["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
