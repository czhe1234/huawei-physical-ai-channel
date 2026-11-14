#!/usr/bin/env python3
"""Train and evaluate the complete Round 2 pre-Sionna channel pipeline."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from carrier_calibration import estimate_carrier
from complex_baselines import predict_one, query_site_neighbors
from data import load_positions
from geometry_features import build_geometry_features
from official_metrics import PointMetricStatistics, aggregate_statistics
from pre_sionna_spectral import (
    M,
    N,
    S,
    build_train_spectral_cache,
    channel_spectra_torch,
    project_channel,
)
from pre_sionna_targets import (
    TargetBundle,
    build_query_spectral_cache,
    build_targets,
)
from project_paths import PATHS
from run_foundation_benchmark import config_templates
from spatial_hole_cv import SpatialHoleFold, build_spatial_hole_folds


ARTIFACT_DIR = PATHS.artifacts / "round2" / "pre_sionna"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_pre_sionna_pipeline"
REPORT = ANALYSIS_DIR / "round2_pre_sionna_pipeline.json"
OOF_BASE = ARTIFACT_DIR / "round2_pre_sionna_oof_foundation.npy"
OOF_META = ARTIFACT_DIR / "round2_pre_sionna_oof_foundation_meta.npz"
ACTION_METRICS = ARTIFACT_DIR / "round2_pre_sionna_oof_action_metrics.npz"
OUTPUT = ARTIFACT_DIR / "Round2_PreSionna_Test_Channel.npy"
SELECTION = ARTIFACT_DIR / "round2_pre_sionna_selection.json"
FOUNDATION_TEST = PATHS.artifacts / "round2" / "foundation" / "Round2_Foundation_Test_Channel.npy"
FOUNDATION_SELECTION = PATHS.artifacts / "round2" / "foundation" / "round2_foundation_selection.json"


@dataclass(frozen=True)
class ActionConfig:
    train_target_mix: float
    alpha_ad: float
    alpha_pas: float
    alpha_pdp: float
    alpha_power: float

    @property
    def name(self) -> str:
        return (
            f"mix{self.train_target_mix:.2f}_ad{self.alpha_ad:.2f}_"
            f"pas{self.alpha_pas:.2f}_pdp{self.alpha_pdp:.2f}_pow{self.alpha_power:.2f}"
        )


@dataclass(frozen=True)
class ActionStatistics:
    pas: np.ndarray
    pdp: np.ndarray
    error: np.ndarray
    energy: np.ndarray
    utility: np.ndarray


def _action_configs() -> list[ActionConfig]:
    spectral = [
        (0.00, 0.00, 0.00),
        (0.25, 0.00, 0.00),
        (0.50, 0.00, 0.00),
        (0.00, 0.30, 0.00),
        (0.00, 0.60, 0.00),
        (0.00, 0.00, 0.35),
        (0.00, 0.00, 0.70),
        (0.00, 0.30, 0.35),
        (0.00, 0.30, 0.70),
        (0.00, 0.60, 0.35),
        (0.00, 0.60, 0.70),
        (0.25, 0.30, 0.35),
        (0.25, 0.30, 0.70),
        (0.25, 0.60, 0.35),
        (0.25, 0.60, 0.70),
        (0.50, 0.30, 0.35),
        (0.50, 0.30, 0.70),
        (0.50, 0.60, 0.35),
        (0.50, 0.60, 0.70),
    ]
    output: list[ActionConfig] = []
    seen: set[tuple[float, float, float, float, float]] = set()
    for mix in (1.00, 0.72):
        for alpha_ad, alpha_pas, alpha_pdp in spectral:
            for alpha_power in (0.00, 0.50, 1.00):
                effective_mix = 1.00 if alpha_ad == alpha_pas == alpha_pdp == 0.0 else mix
                key = (effective_mix, alpha_ad, alpha_pas, alpha_pdp, alpha_power)
                if key in seen:
                    continue
                seen.add(key)
                output.append(ActionConfig(*key))
    return output


def _load_site_configs() -> dict[int, object]:
    payload = json.loads(FOUNDATION_SELECTION.read_text(encoding="utf-8"))
    names = {int(site): name for site, name in payload["site_configs"].items()}
    templates = {config.name: config for config in config_templates()}
    return {site: templates[name] for site, name in names.items()}


def _query_rows_by_fold(query_ids: np.ndarray, folds: list[SpatialHoleFold]) -> np.ndarray:
    global_to_row = np.full(int(np.max(query_ids)) + 1, -1, dtype=np.int64)
    global_to_row[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    fold_id = np.full(len(query_ids), -1, dtype=np.int8)
    for fold in folds:
        fold_id[global_to_row[fold.query_indices]] = fold.fold
    if np.any(fold_id < 0):
        raise RuntimeError("not every OOF query was assigned to a fold")
    return fold_id


def generate_oof_foundation(
    train_channel: np.ndarray,
    train_positions: np.ndarray,
    train_site: np.ndarray,
    bs_positions: np.ndarray,
    point_power: np.ndarray,
    folds: list[SpatialHoleFold],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_ids = np.sort(np.concatenate([fold.query_indices for fold in folds])).astype(np.int64)
    fold_id = _query_rows_by_fold(query_ids, folds)
    if OOF_BASE.is_file() and OOF_META.is_file():
        base = np.load(OOF_BASE, mmap_mode="r")
        meta = np.load(OOF_META)
        if base.shape == (len(query_ids), M, N, S) and np.array_equal(meta["query_ids"], query_ids):
            return base, query_ids, meta["fold_id"].astype(np.int8)

    configs = _load_site_configs()
    global_to_row = np.full(len(train_positions), -1, dtype=np.int64)
    global_to_row[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    output = np.lib.format.open_memmap(
        OOF_BASE, mode="w+", dtype=np.complex64, shape=(len(query_ids), M, N, S)
    )
    for fold in folds:
        carriers = estimate_carrier(
            train_channel,
            train_positions,
            train_site,
            bs_positions,
            fold.fit_indices,
            point_power,
            max_pairs_per_site=500,
            seed=20_260_805 + fold.fold * 101,
        )
        distance, neighbors = query_site_neighbors(
            train_positions,
            train_site,
            fold.fit_indices,
            fold.query_indices,
            max_k=max(config.k for config in configs.values()),
        )
        for local, query_id in enumerate(fold.query_indices):
            site = int(train_site[query_id])
            config = replace(
                configs[site], beta=carriers[site].beta, gamma=carriers[site].gamma
            )
            neighbor_ids = neighbors[local]
            prediction, _ = predict_one(
                np.asarray(train_channel[neighbor_ids], dtype=np.complex64),
                train_positions[neighbor_ids],
                train_positions[query_id],
                distance[local],
                bs_positions[site],
                config,
            )
            output[global_to_row[query_id]] = prediction
            if (local + 1) % 10 == 0 or local + 1 == len(fold.query_indices):
                output.flush()
                print(
                    f"pre-Sionna OOF foundation fold={fold.fold} "
                    f"queries={local + 1}/{len(fold.query_indices)}",
                    flush=True,
                )
    np.savez_compressed(OOF_META, query_ids=query_ids, fold_id=fold_id)
    return np.load(OOF_BASE, mmap_mode="r"), query_ids, fold_id


def _spectral_point_scores(
    prediction: torch.Tensor,
    truth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, pred_pas, pred_pdp = channel_spectra_torch(prediction)
    _, truth_pas, truth_pdp = channel_spectra_torch(truth)
    pas_num = (pred_pas * truth_pas).sum(dim=1)
    pas_den = torch.sqrt(
        pred_pas.square().sum(dim=1) * truth_pas.square().sum(dim=1)
    )
    pas_den = torch.where(pas_den == 0.0, torch.full_like(pas_den, 1e-12), pas_den)
    pas = (pas_num / pas_den).mean(dim=(1, 2))
    pdp_num = (pred_pdp * truth_pdp).sum(dim=3)
    pdp_den = torch.sqrt(
        pred_pdp.square().sum(dim=3) * truth_pdp.square().sum(dim=3)
    )
    pdp_den = torch.where(pdp_den == 0.0, torch.full_like(pdp_den, 1e-12), pdp_den)
    pdp = (pdp_num / pdp_den).mean(dim=(1, 2))
    pred_energy = prediction.abs().square().sum(dim=(1, 2, 3)).double()
    truth_energy = truth.abs().square().sum(dim=(1, 2, 3)).double()
    cross = torch.real(torch.sum(torch.conj(prediction) * truth, dim=(1, 2, 3))).double()
    return pas.double(), pdp.double(), pred_energy, truth_energy, cross


def evaluate_action_grid(
    base: np.ndarray,
    truth: np.ndarray,
    targets: TargetBundle,
    batch_size: int = 4,
) -> tuple[list[ActionConfig], ActionStatistics]:
    configs = _action_configs()
    if ACTION_METRICS.is_file():
        cached = np.load(ACTION_METRICS, allow_pickle=False)
        cached_names = cached["action_names"].astype(str).tolist()
        if cached_names == [config.name for config in configs] and cached["pas"].shape == (len(configs), len(base)):
            return configs, ActionStatistics(
                pas=cached["pas"].astype(np.float64),
                pdp=cached["pdp"].astype(np.float64),
                error=cached["error"].astype(np.float64),
                energy=cached["energy"].astype(np.float64),
                utility=cached["utility"].astype(np.float64),
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spectral_keys = list(
        dict.fromkeys(
            (config.train_target_mix, config.alpha_ad, config.alpha_pas, config.alpha_pdp)
            for config in configs
        )
    )
    spectral_index = {key: slot for slot, key in enumerate(spectral_keys)}
    pas = np.zeros((len(configs), len(base)), dtype=np.float64)
    pdp = np.zeros_like(pas)
    error = np.zeros_like(pas)
    energy = np.zeros_like(pas)

    base_log_power = np.log(np.maximum(targets.metadata.get("base_power", np.ones(len(base))), 1e-30))
    if "base_power" not in targets.metadata:
        base_power = np.empty(len(base), dtype=np.float64)
        for start in range(0, len(base), 16):
            end = min(start + 16, len(base))
            block = np.asarray(base[start:end], dtype=np.complex64)
            base_power[start:end] = np.sum(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64)
        base_log_power = np.log(np.maximum(base_power, 1e-30))
    log_delta = np.clip(targets.predicted_log_power - base_log_power, -2.5, 2.5)

    actions_by_spectral: dict[int, list[tuple[int, ActionConfig]]] = {}
    for action_index, config in enumerate(configs):
        key = (config.train_target_mix, config.alpha_ad, config.alpha_pas, config.alpha_pdp)
        actions_by_spectral.setdefault(spectral_index[key], []).append((action_index, config))

    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        base_tensor = torch.as_tensor(
            np.asarray(base[start:end], dtype=np.complex64), device=device
        )
        truth_tensor = torch.as_tensor(
            np.asarray(truth[start:end], dtype=np.complex64), device=device
        )
        train_ad = torch.as_tensor(np.asarray(targets.train_ad[start:end], dtype=np.float32), device=device)
        train_pas = torch.as_tensor(np.asarray(targets.train_pas[start:end], dtype=np.float32), device=device)
        train_pdp = torch.as_tensor(np.asarray(targets.train_pdp[start:end], dtype=np.float32), device=device)
        consensus_ad = torch.as_tensor(
            np.asarray(targets.consensus_ad[start:end], dtype=np.float32), device=device
        )
        consensus_pas = torch.as_tensor(
            np.asarray(targets.consensus_pas[start:end], dtype=np.float32), device=device
        )
        consensus_pdp = torch.as_tensor(
            np.asarray(targets.consensus_pdp[start:end], dtype=np.float32), device=device
        )
        with torch.inference_mode():
            for key_index, key in enumerate(spectral_keys):
                mix, alpha_ad, alpha_pas, alpha_pdp = key
                ad_target = mix * train_ad + (1.0 - mix) * consensus_ad
                pas_target = mix * train_pas + (1.0 - mix) * consensus_pas
                pdp_target = mix * train_pdp + (1.0 - mix) * consensus_pdp
                prediction = project_channel(
                    base_tensor,
                    ad_target,
                    pas_target,
                    pdp_target,
                    alpha_ad,
                    alpha_pas,
                    alpha_pdp,
                )
                point_pas, point_pdp, pred_energy, truth_energy, cross = _spectral_point_scores(
                    prediction, truth_tensor
                )
                for action_index, config in actions_by_spectral[key_index]:
                    scale = torch.as_tensor(
                        np.exp(0.5 * config.alpha_power * log_delta[start:end]),
                        dtype=torch.float64,
                        device=device,
                    )
                    point_error = (
                        scale.square() * pred_energy + truth_energy - 2.0 * scale * cross
                    ).clamp_min(0.0)
                    pas[action_index, start:end] = point_pas.cpu().numpy()
                    pdp[action_index, start:end] = point_pdp.cpu().numpy()
                    error[action_index, start:end] = point_error.cpu().numpy()
                    energy[action_index, start:end] = truth_energy.cpu().numpy()
        print(f"action grid {end}/{len(base)}", flush=True)

    nmse_score = 1.0 / (1.0 + error / np.maximum(energy, 1e-30))
    utility = 0.4 * pas + 0.4 * pdp + 0.2 * nmse_score
    np.savez_compressed(
        ACTION_METRICS,
        action_names=np.asarray([config.name for config in configs], dtype="U80"),
        pas=pas,
        pdp=pdp,
        error=error,
        energy=energy,
        utility=utility,
    )
    return configs, ActionStatistics(pas, pdp, error, energy, utility)


def _aggregate_arrays(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
    rows: np.ndarray,
) -> dict[str, float | int]:
    return aggregate_statistics(
        PointMetricStatistics(
            pas=np.asarray(pas[rows], dtype=np.float64),
            pdp=np.asarray(pdp[rows], dtype=np.float64),
            squared_error=np.asarray(error[rows], dtype=np.float64),
            truth_energy=np.asarray(energy[rows], dtype=np.float64),
        )
    )


def _action_summary(stats: ActionStatistics, action_index: int, rows: np.ndarray) -> dict[str, float | int]:
    return _aggregate_arrays(
        stats.pas[action_index],
        stats.pdp[action_index],
        stats.error[action_index],
        stats.energy[action_index],
        rows,
    )


def _zero_replaced(
    stats: ActionStatistics,
    action_index: int,
    zero_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pas = stats.pas[action_index].copy()
    pdp = stats.pdp[action_index].copy()
    error = stats.error[action_index].copy()
    energy = stats.energy[action_index].copy()
    pas[zero_mask] = 0.0
    pdp[zero_mask] = 0.0
    error[zero_mask] = energy[zero_mask]
    return pas, pdp, error, energy


def _select_action(
    stats: ActionStatistics,
    configs: list[ActionConfig],
    train_rows: np.ndarray,
    fold_id: np.ndarray,
    base_index: int,
    min_positive_folds: int,
) -> tuple[int, list[dict[str, float]]]:
    folds = np.unique(fold_id[train_rows])
    base_scores = {
        int(fold): float(_action_summary(stats, base_index, train_rows[fold_id[train_rows] == fold])["score"])
        for fold in folds
    }
    candidates: list[tuple[int, int, float, list[float]]] = []
    for action_index in range(len(configs)):
        deltas = []
        for fold in folds:
            rows = train_rows[fold_id[train_rows] == fold]
            score = float(_action_summary(stats, action_index, rows)["score"])
            deltas.append(score - base_scores[int(fold)])
        positive = int(np.sum(np.asarray(deltas) > 0.0))
        candidates.append((action_index, positive, float(np.mean(deltas)), deltas))
    eligible = [item for item in candidates if item[0] == base_index or item[1] >= min_positive_folds]
    selected = max(eligible, key=lambda item: item[2])
    rows = [
        {
            "action_index": int(index),
            "positive_folds": int(positive),
            "mean_fold_delta": float(mean_delta),
        }
        for index, positive, mean_delta, _ in sorted(candidates, key=lambda item: item[2], reverse=True)[:12]
    ]
    return int(selected[0]), rows


def _select_zero_threshold(
    stats: ActionStatistics,
    action_index: int,
    train_rows: np.ndarray,
    fold_id: np.ndarray,
    zero_probability: np.ndarray,
    min_positive_folds: int,
) -> float:
    thresholds = (1.10, 0.65, 0.80, 0.90)
    folds = np.unique(fold_id[train_rows])
    baseline = {}
    for fold in folds:
        rows = train_rows[fold_id[train_rows] == fold]
        baseline[int(fold)] = float(_action_summary(stats, action_index, rows)["score"])
    candidates = []
    for threshold in thresholds:
        zero_mask = zero_probability >= threshold
        pas, pdp, error, energy = _zero_replaced(stats, action_index, zero_mask)
        deltas = []
        for fold in folds:
            rows = train_rows[fold_id[train_rows] == fold]
            score = float(_aggregate_arrays(pas, pdp, error, energy, rows)["score"])
            deltas.append(score - baseline[int(fold)])
        candidates.append((threshold, int(np.sum(np.asarray(deltas) > 0.0)), float(np.mean(deltas))))
    eligible = [item for item in candidates if item[0] > 1.0 or item[1] >= min_positive_folds]
    return float(max(eligible, key=lambda item: item[2])[0])


def nested_controller(
    stats: ActionStatistics,
    configs: list[ActionConfig],
    features: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
    zero_probability: np.ndarray,
) -> dict[str, object]:
    base_index = next(
        index
        for index, config in enumerate(configs)
        if config.alpha_ad == config.alpha_pas == config.alpha_pdp == config.alpha_power == 0.0
    )
    clean_features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
    fixed_action = np.full(len(site), base_index, dtype=np.int32)
    controlled_action = np.full(len(site), base_index, dtype=np.int32)
    fixed_zero = np.zeros(len(site), dtype=bool)
    controlled_zero = np.zeros(len(site), dtype=bool)
    decisions: list[dict[str, object]] = []

    for held_fold in np.unique(fold_id):
        for site_value in np.unique(site):
            held = np.flatnonzero((fold_id == held_fold) & (site == site_value))
            train = np.flatnonzero((fold_id != held_fold) & (site == site_value))
            action_index, ranking = _select_action(
                stats,
                configs,
                train,
                fold_id,
                base_index,
                min_positive_folds=3,
            )
            zero_threshold = _select_zero_threshold(
                stats,
                action_index,
                train,
                fold_id,
                zero_probability,
                min_positive_folds=3,
            )
            fixed_action[held] = action_index
            fixed_zero[held] = zero_probability[held] >= zero_threshold
            gain = stats.utility[action_index, train] - stats.utility[base_index, train]
            model = ExtraTreesRegressor(
                n_estimators=360,
                min_samples_leaf=9,
                max_features=0.82,
                random_state=20_260_805 + 101 * int(held_fold) + int(site_value),
                n_jobs=-1,
            )
            model.fit(clean_features[train], gain)
            predicted_gain = model.predict(clean_features[held])
            active = predicted_gain > 0.0
            controlled_action[held[active]] = action_index
            controlled_zero[held] = zero_probability[held] >= zero_threshold
            decisions.append(
                {
                    "held_fold": int(held_fold),
                    "site": int(site_value),
                    "train_count": int(len(train)),
                    "held_count": int(len(held)),
                    "selected_action": configs[action_index].name,
                    "zero_threshold": zero_threshold,
                    "controller_active": int(np.sum(active)),
                    "top_training_actions": ranking,
                }
            )

    def selected_statistics(action: np.ndarray, zero: np.ndarray) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        pas = stats.pas[base_index].copy()
        pdp = stats.pdp[base_index].copy()
        error = stats.error[base_index].copy()
        energy = stats.energy[base_index].copy()
        for action_index in np.unique(action):
            rows = action == action_index
            pas[rows] = stats.pas[action_index, rows]
            pdp[rows] = stats.pdp[action_index, rows]
            error[rows] = stats.error[action_index, rows]
        pas[zero] = 0.0
        pdp[zero] = 0.0
        error[zero] = energy[zero]
        all_rows = np.arange(len(site))
        report = {
            "all": _aggregate_arrays(pas, pdp, error, energy, all_rows),
            "folds": [
                {
                    "fold": int(fold),
                    **_aggregate_arrays(pas, pdp, error, energy, np.flatnonzero(fold_id == fold)),
                }
                for fold in np.unique(fold_id)
            ],
            "sites": {
                str(int(site_value)): _aggregate_arrays(
                    pas, pdp, error, energy, np.flatnonzero(site == site_value)
                )
                for site_value in np.unique(site)
            },
            "active_count": int(np.sum(action != base_index)),
            "zero_count": int(np.sum(zero)),
        }
        return report, {"pas": pas, "pdp": pdp, "error": error, "energy": energy}

    fixed_report, fixed_values = selected_statistics(fixed_action, fixed_zero)
    controlled_report, controlled_values = selected_statistics(controlled_action, controlled_zero)
    base_report = _action_summary(stats, base_index, np.arange(len(site)))
    mode = "controlled" if float(controlled_report["all"]["score"]) >= float(fixed_report["all"]["score"]) else "fixed"
    return {
        "base_index": base_index,
        "base": base_report,
        "fixed": fixed_report,
        "controlled": controlled_report,
        "selected_mode": mode,
        "selected_action": controlled_action if mode == "controlled" else fixed_action,
        "selected_zero": controlled_zero if mode == "controlled" else fixed_zero,
        "selected_values": controlled_values if mode == "controlled" else fixed_values,
        "decisions": decisions,
    }


def final_models(
    stats: ActionStatistics,
    configs: list[ActionConfig],
    oof_features: np.ndarray,
    fold_id: np.ndarray,
    oof_site: np.ndarray,
    oof_zero_probability: np.ndarray,
    test_features: np.ndarray,
    test_site: np.ndarray,
    test_zero_probability: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    base_index = next(
        index
        for index, config in enumerate(configs)
        if config.alpha_ad == config.alpha_pas == config.alpha_pdp == config.alpha_power == 0.0
    )
    clean_oof = np.nan_to_num(oof_features, nan=0.0, posinf=100.0, neginf=-100.0)
    clean_test = np.nan_to_num(test_features, nan=0.0, posinf=100.0, neginf=-100.0)
    test_action = np.full(len(test_site), base_index, dtype=np.int32)
    test_zero = np.zeros(len(test_site), dtype=bool)
    records: list[dict[str, object]] = []
    for site_value in np.unique(oof_site):
        train = np.flatnonzero(oof_site == site_value)
        test = np.flatnonzero(test_site == site_value)
        action_index, ranking = _select_action(
            stats,
            configs,
            train,
            fold_id,
            base_index,
            min_positive_folds=4,
        )
        zero_threshold = _select_zero_threshold(
            stats,
            action_index,
            train,
            fold_id,
            oof_zero_probability,
            min_positive_folds=4,
        )
        active = np.ones(len(test), dtype=bool)
        if mode == "controlled" and len(test) > 0:
            gain = stats.utility[action_index, train] - stats.utility[base_index, train]
            model = ExtraTreesRegressor(
                n_estimators=480,
                min_samples_leaf=9,
                max_features=0.82,
                random_state=20_260_805 + int(site_value),
                n_jobs=-1,
            )
            model.fit(clean_oof[train], gain)
            active = model.predict(clean_test[test]) > 0.0
        test_action[test[active]] = action_index
        test_zero[test] = test_zero_probability[test] >= zero_threshold
        records.append(
            {
                "site": int(site_value),
                "selected_action": configs[action_index].name,
                "mode": mode,
                "active_count": int(np.sum(active)),
                "test_count": int(len(test)),
                "zero_threshold": zero_threshold,
                "zero_count": int(np.sum(test_zero[test])),
                "top_oof_actions": ranking,
            }
        )
    return test_action, test_zero, records


def generate_output(
    base: np.ndarray,
    targets: TargetBundle,
    configs: list[ActionConfig],
    actions: np.ndarray,
    zero: np.ndarray,
    batch_size: int = 4,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for action_index in np.unique(actions):
        rows = np.flatnonzero((actions == action_index) & ~zero)
        config = configs[int(action_index)]
        for start in range(0, len(rows), batch_size):
            selected = rows[start : start + batch_size]
            channel = torch.as_tensor(np.asarray(base[selected], dtype=np.complex64), device=device)
            train_ad = torch.as_tensor(np.asarray(targets.train_ad[selected], dtype=np.float32), device=device)
            train_pas = torch.as_tensor(np.asarray(targets.train_pas[selected], dtype=np.float32), device=device)
            train_pdp = torch.as_tensor(np.asarray(targets.train_pdp[selected], dtype=np.float32), device=device)
            consensus_ad = torch.as_tensor(
                np.asarray(targets.consensus_ad[selected], dtype=np.float32), device=device
            )
            consensus_pas = torch.as_tensor(
                np.asarray(targets.consensus_pas[selected], dtype=np.float32), device=device
            )
            consensus_pdp = torch.as_tensor(
                np.asarray(targets.consensus_pdp[selected], dtype=np.float32), device=device
            )
            mix = config.train_target_mix
            with torch.inference_mode():
                prediction = project_channel(
                    channel,
                    mix * train_ad + (1.0 - mix) * consensus_ad,
                    mix * train_pas + (1.0 - mix) * consensus_pas,
                    mix * train_pdp + (1.0 - mix) * consensus_pdp,
                    config.alpha_ad,
                    config.alpha_pas,
                    config.alpha_pdp,
                )
                base_power = channel.abs().square().sum(dim=(1, 2, 3)).double()
                log_base = torch.log(base_power.clamp_min(1e-30))
                log_target = torch.as_tensor(
                    targets.predicted_log_power[selected], dtype=torch.float64, device=device
                )
                log_delta = (log_target - log_base).clamp(-2.5, 2.5)
                scale = torch.exp(0.5 * config.alpha_power * log_delta)
                prediction = prediction * scale[:, None, None, None]
            output[selected] = prediction.cpu().numpy().astype(np.complex64)
            print(
                f"test output action={config.name} rows={start + len(selected)}/{len(rows)}",
                flush=True,
            )
        output.flush()
    if np.any(zero):
        output[zero] = np.complex64(0.0)
        output.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    setup, train_positions, test_positions, train_site, test_site = load_positions()
    train_channel = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    geometry = build_geometry_features(
        train_positions,
        test_positions,
        train_site,
        test_site,
        setup.bs_positions,
    )
    spectral = build_train_spectral_cache(train_channel, batch_size=args.batch_size)
    folds = build_spatial_hole_folds(train_positions, test_positions, train_site, test_site)
    oof_base, query_ids, fold_id = generate_oof_foundation(
        train_channel,
        train_positions,
        train_site,
        setup.bs_positions,
        spectral.total_power,
        folds,
    )
    oof_truth = np.asarray(train_channel[query_ids], dtype=np.complex64)
    oof_spectral = build_query_spectral_cache(
        oof_base, ARTIFACT_DIR, "oof_foundation", batch_size=args.batch_size
    )
    global_to_row = np.full(len(train_positions), -1, dtype=np.int64)
    global_to_row[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    fit_groups = [
        (fold.fold, fold.fit_indices, global_to_row[fold.query_indices]) for fold in folds
    ]
    oof_targets = build_targets(
        train_positions,
        train_positions[query_ids],
        train_site,
        train_site[query_ids],
        geometry.train,
        geometry.train[query_ids],
        spectral,
        oof_spectral,
        fit_groups,
        ARTIFACT_DIR,
        "oof",
    )
    configs, action_stats = evaluate_action_grid(
        oof_base, oof_truth, oof_targets, batch_size=args.batch_size
    )
    controller = nested_controller(
        action_stats,
        configs,
        oof_targets.controller_features,
        fold_id,
        train_site[query_ids],
        oof_targets.zero_probability,
    )

    test_base = np.load(FOUNDATION_TEST, mmap_mode="r")
    test_spectral = build_query_spectral_cache(
        test_base, ARTIFACT_DIR, "test_foundation", batch_size=args.batch_size
    )
    test_targets = build_targets(
        train_positions,
        test_positions,
        train_site,
        test_site,
        geometry.train,
        geometry.test,
        spectral,
        test_spectral,
        [(99, np.arange(len(train_positions), dtype=np.int64), np.arange(len(test_positions), dtype=np.int64))],
        ARTIFACT_DIR,
        "test",
    )
    test_action, test_zero, final_records = final_models(
        action_stats,
        configs,
        oof_targets.controller_features,
        fold_id,
        train_site[query_ids],
        oof_targets.zero_probability,
        test_targets.controller_features,
        test_site,
        test_targets.zero_probability,
        str(controller["selected_mode"]),
    )
    generate_output(
        test_base,
        test_targets,
        configs,
        test_action,
        test_zero,
        batch_size=args.batch_size,
    )

    output = np.load(OUTPUT, mmap_mode="r")
    finite = True
    for start in range(0, len(output), 16):
        block = np.asarray(output[start : start + 16], dtype=np.complex64)
        finite = finite and bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
    if output.shape != (setup.test_count, setup.tx_antennas, setup.rx_antennas, setup.subcarriers):
        raise RuntimeError(f"invalid output shape: {output.shape}")
    if output.dtype != np.complex64 or not finite:
        raise RuntimeError(f"invalid output dtype/finite: {output.dtype}/{finite}")

    base_score = controller["base"]
    fixed_score = controller["fixed"]
    controlled_score = controller["controlled"]
    selected_score = controller[str(controller["selected_mode"])]
    report = {
        "experiment": "Round2-E2 complete pre-Sionna pipeline",
        "status": "analysis_foundation_no_submission_archive",
        "primary_hypothesis": (
            "Fit-only propagation-state and visibility conditioned angle-delay/PAS/PDP templates, "
            "followed by an OOF-trained point controller and power calibration, should improve the "
            "site-aware complex foundation without any Sionna channel information."
        ),
        "sionna_used": False,
        "round1_prediction_cache_used": False,
        "official_inputs_only": [
            "Round2_Train_Channel.npy",
            "Round2_Train_Pos.npy",
            "Round2_Test_Pos.npy",
            "Round2_Map.ply",
            "Round2_Setup.json",
        ],
        "modules": {
            "angle_delay_support_template": "fit-only joint 16x8x192 support and 12-state site template model",
            "propagation_visibility": "official-map five-ray fan, local surface features, zero-state and template probabilities",
            "pas_pdp_objectives": "exact official per-link PAS and PDP normalized targets",
            "power_and_point_control": "fit-only log-power ensemble, zero probability and fold-held-out ExtraTrees gain controller",
            "adaptive_projection": "angle-delay then PAS then PDP amplitude projection while preserving foundation phase and point power before calibrated scaling",
        },
        "validation": {
            "method": "five spatial holes matching the ten official test clusters",
            "query_count": int(len(query_ids)),
            "base": base_score,
            "fixed_nested": fixed_score,
            "controlled_nested": controlled_score,
            "selected_mode": controller["selected_mode"],
            "selected": selected_score,
            "decisions": controller["decisions"],
        },
        "final_test_policy": final_records,
        "action_count": len(configs),
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "selection": PATHS.relative(SELECTION),
            "action_metrics": PATHS.relative(ACTION_METRICS),
            "report": PATHS.relative(REPORT),
        },
        "integrity": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": finite,
            "bytes": int(OUTPUT.stat().st_size),
        },
        "seconds": float(time.time() - started),
    }
    serializable_selection = {
        "selected_mode": controller["selected_mode"],
        "oof_base_score": base_score,
        "oof_selected_score": selected_score,
        "test_actions": [configs[int(index)].name for index in test_action],
        "test_zero": test_zero.astype(int).tolist(),
        "final_test_policy": final_records,
    }
    SELECTION.write_text(
        json.dumps(serializable_selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "base": base_score,
                "fixed": fixed_score,
                "controlled": controlled_score,
                "selected_mode": controller["selected_mode"],
                "final_test_policy": final_records,
                "output": PATHS.relative(OUTPUT),
                "seconds": report["seconds"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
