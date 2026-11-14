#!/usr/bin/env python3
"""Inject deployable wall phase before the validated V4 spectral projection."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from analyze_v4_exact_joint_projection import continue_trajectory, initialize_trajectory
from analyze_v5_deployable_wall_phase_transport import (
    PHASE_CACHE,
    gate_masks,
    mix_full_phase,
)
from analyze_v5_path_relative_port_phase import (
    fixed_coordinate_decomposition,
    ports_to_channel,
    preserve_power,
)
from analyze_v5_static_link_phase_calibration import (
    TRAIN_CHANNEL,
    V4_OOF_CHANNEL,
    V4_REPORT,
    parse_candidate,
)
from evaluate_v3_anchored_projection import target_batch
from official_metrics import point_metric_statistics
from pre_sionna_spectral import project_angle_delay
from project_paths import PATHS
from run_rbf_spectral_field import ACTIONS, OOF_BASE, OOF_META, statistics
from run_v4_calibrated_exact_projection import (
    OOF_CALIBRATED_PAS,
    OOF_CALIBRATED_PDP,
    TARGET_AD,
    TARGET_PAS,
    TARGET_PDP,
    mode_targets,
)


VERSION = "Round2-E34-preprojection-wall-phase"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v5_preprojection_wall_phase"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v5_preprojection_wall_phase"
METRIC_CACHE = ARTIFACT_DIR / "oof_preprojection_wall_phase_metrics.npz"
REPORT = ANALYSIS_DIR / "preprojection_wall_phase.json"


@dataclass(frozen=True)
class Config:
    gate: str = "quality_top75"
    phase_alpha: float = 0.10
    decomposition_ridge: float = 0.003
    deployment_score_threshold: float = 0.645
    deployment_positive_folds: int = 4


VARIANT_SOURCES = {
    "correct_geometry": "source",
    "rotated_geometry_control": "rotated_source",
    "wrong_wavenumber_control": "wrong_beta_source",
}
VARIANT_NAMES = ("identity", *VARIANT_SOURCES)


def load_v4_decisions() -> dict[tuple[int, int], dict[str, object]]:
    payload = json.loads(V4_REPORT.read_text(encoding="utf-8"))
    return {
        (int(row["held_fold"]), int(row["site"])): row
        for row in payload["nested_selected"]["decisions"]
    }


@torch.inference_mode()
def replay_v4(
    base: torch.Tensor,
    target_base: torch.Tensor,
    ad: torch.Tensor,
    raw_pas: torch.Tensor,
    raw_pdp: torch.Tensor,
    calibrated_pas: torch.Tensor,
    calibrated_pdp: torch.Tensor,
    local_fold: np.ndarray,
    local_site: np.ndarray,
    decisions: dict[tuple[int, int], dict[str, object]],
) -> torch.Tensor:
    output = torch.empty_like(base)
    for fold_value in np.unique(local_fold):
        for site_value in np.unique(local_site[local_fold == fold_value]):
            rows = np.flatnonzero(
                (local_fold == fold_value) & (local_site == site_value)
            )
            row_tensor = torch.as_tensor(rows, dtype=torch.long, device=base.device)
            decision = decisions[(int(fold_value), int(site_value))]
            mode, order, action_index, cycles = parse_candidate(
                str(decision["candidate"])
            )
            pas, pdp = mode_targets(
                mode,
                raw_pas[row_tensor],
                raw_pdp[row_tensor],
                calibrated_pas[row_tensor],
                calibrated_pdp[row_tensor],
            )
            (
                target_candidate,
                desired_pas,
                desired_pdp,
                target_reference_power,
            ) = initialize_trajectory(
                target_base[row_tensor],
                ad[row_tensor],
                pas,
                pdp,
                action_index,
                order,
            )
            if base.data_ptr() == target_base.data_ptr():
                candidate = target_candidate
                reference_power = target_reference_power
            else:
                action = ACTIONS[action_index]
                reference_power = base[row_tensor].abs().square().sum(dim=(1, 2, 3))
                candidate = project_angle_delay(
                    base[row_tensor], ad[row_tensor], action.alpha_ad
                )
                candidate = continue_trajectory(
                    candidate,
                    desired_pas,
                    desired_pdp,
                    reference_power,
                    order,
                )
            for _ in range(2, cycles + 1):
                candidate = continue_trajectory(
                    candidate,
                    desired_pas,
                    desired_pdp,
                    reference_power,
                    order,
                )
            output[row_tensor] = candidate * float(decision["scale"])
    return output


def candidate_statistics(
    names: tuple[str, ...],
    metric: dict[str, np.ndarray],
    fold_id: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, name in enumerate(names):
        overall = statistics(
            metric["pas"][index],
            metric["pdp"][index],
            metric["error"][index],
            metric["energy"][index],
        )
        rows.append(
            {
                "name": name,
                **overall,
                "folds": [
                    {
                        "fold": int(fold),
                        **statistics(
                            metric["pas"][index],
                            metric["pdp"][index],
                            metric["error"][index],
                            metric["energy"][index],
                            fold_id == fold,
                        ),
                    }
                    for fold in np.unique(fold_id)
                ],
            }
        )
    return rows


def nested_select_real_action(
    metric: dict[str, np.ndarray],
    fold_id: np.ndarray,
    site: np.ndarray,
) -> dict[str, object]:
    selectable_names = ("identity", "correct_geometry")
    selected = np.zeros(len(fold_id), dtype=np.int64)
    decisions: list[dict[str, object]] = []
    for held_fold in np.unique(fold_id):
        for site_value in np.unique(site):
            train = (fold_id != held_fold) & (site == site_value)
            held = (fold_id == held_fold) & (site == site_value)
            scores = np.asarray(
                [
                    statistics(
                        metric["pas"][index],
                        metric["pdp"][index],
                        metric["error"][index],
                        metric["energy"][index],
                        train,
                    )["score"]
                    for index in range(len(selectable_names))
                ],
                dtype=np.float64,
            )
            chosen = int(np.argmax(scores))
            selected[held] = chosen
            decisions.append(
                {
                    "held_fold": int(held_fold),
                    "site": int(site_value),
                    "candidate": selectable_names[chosen],
                    "training_score": float(scores[chosen]),
                }
            )

    point = np.arange(len(fold_id))
    result = statistics(
        metric["pas"][selected, point],
        metric["pdp"][selected, point],
        metric["error"][selected, point],
        metric["energy"][selected, point],
    )
    baseline = statistics(
        metric["pas"][0],
        metric["pdp"][0],
        metric["error"][0],
        metric["energy"][0],
    )
    fold_delta: list[float] = []
    folds: list[dict[str, object]] = []
    for fold in np.unique(fold_id):
        mask = fold_id == fold
        current = statistics(
            metric["pas"][selected, point],
            metric["pdp"][selected, point],
            metric["error"][selected, point],
            metric["energy"][selected, point],
            mask,
        )
        old = statistics(
            metric["pas"][0],
            metric["pdp"][0],
            metric["error"][0],
            metric["energy"][0],
            mask,
        )
        delta = float(current["score"] - old["score"])
        fold_delta.append(delta)
        folds.append({"fold": int(fold), **current, "delta_vs_identity": delta})
    result["delta_vs_identity"] = float(result["score"] - baseline["score"])
    result["fold_delta_vs_identity"] = fold_delta
    result["positive_fold_count"] = int(np.sum(np.asarray(fold_delta) > 0.0))
    result["folds"] = folds
    result["decisions"] = decisions
    result["selectable_candidates"] = list(selectable_names)
    return result


@torch.inference_mode()
def evaluate(
    arrays: dict[str, np.ndarray],
    config: Config,
    device: torch.device,
    batch_size: int,
    rebuild: bool,
) -> tuple[dict[str, object], dict[str, float]]:
    gate_names, all_masks = gate_masks(arrays)
    gate_index = gate_names.index(config.gate)
    phase_mask = all_masks[gate_index]
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    site = arrays["site"].astype(np.int8)
    if not np.array_equal(query_ids, arrays["query_ids"].astype(np.int64)):
        raise RuntimeError("phase cache and V4 OOF query order differs")
    if not np.array_equal(fold_id, arrays["fold_id"].astype(np.int8)):
        raise RuntimeError("phase cache and V4 OOF fold order differs")

    if METRIC_CACHE.is_file() and not rebuild:
        cached = np.load(METRIC_CACHE)
        cached_names = tuple(str(value) for value in cached["variant_names"])
        if cached_names != VARIANT_NAMES:
            raise RuntimeError("cached E34 variants do not match current configuration")
        metric = {key: cached[key] for key in ("pas", "pdp", "error", "energy")}
        diagnostics = json.loads(str(cached["diagnostics_json"].item()))
        return summarize(metric, fold_id, site, config), diagnostics

    base_all = np.load(OOF_BASE, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    v4_reference = np.load(V4_OOF_CHANNEL, mmap_mode="r")
    ad_cache = np.load(TARGET_AD, mmap_mode="r")
    pas_cache = np.load(TARGET_PAS, mmap_mode="r")
    pdp_cache = np.load(TARGET_PDP, mmap_mode="r")
    calibrated_pas_cache = np.load(OOF_CALIBRATED_PAS, mmap_mode="r")
    calibrated_pdp_cache = np.load(OOF_CALIBRATED_PDP, mmap_mode="r")
    decisions = load_v4_decisions()
    metric = {
        key: np.zeros((len(VARIANT_NAMES), len(query_ids)), dtype=np.float64)
        for key in ("pas", "pdp", "error", "energy")
    }
    identity_difference_energy = 0.0
    identity_reference_energy = 0.0
    identity_max_absolute = 0.0
    residual_energy = 0.0
    base_energy = 0.0

    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        ad, raw_pas, raw_pdp = target_batch(
            ad_cache, pas_cache, pdp_cache, start, end, device
        )
        calibrated_pas = torch.as_tensor(
            np.asarray(calibrated_pas_cache[start:end], dtype=np.float32), device=device
        )
        calibrated_pdp = torch.as_tensor(
            np.asarray(calibrated_pdp_cache[start:end], dtype=np.float32), device=device
        )
        base = torch.as_tensor(
            np.array(base_all[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        coefficient, residual, atoms = fixed_coordinate_decomposition(
            base,
            arrays["coordinates"][start:end],
            config.decomposition_ridge,
            device,
        )
        residual_energy += float(residual.abs().square().sum(dtype=torch.float64).item())
        base_energy += float(base.abs().square().sum(dtype=torch.float64).item())

        preprojection: dict[str, torch.Tensor] = {"identity": base}
        mask = torch.as_tensor(phase_mask[start:end], device=device)
        for variant, source_key in VARIANT_SOURCES.items():
            source = torch.as_tensor(arrays[source_key][start:end], device=device)
            modified = mix_full_phase(coefficient, source, mask, config.phase_alpha)
            ports = residual + torch.einsum("bkc,bkhvs->bchvs", modified, atoms)
            preprojection[variant] = preserve_power(ports_to_channel(ports), base)

        truth_numpy = truth.cpu().numpy()
        for index, name in enumerate(VARIANT_NAMES):
            prediction = replay_v4(
                preprojection[name],
                base,
                ad,
                raw_pas,
                raw_pdp,
                calibrated_pas,
                calibrated_pdp,
                fold_id[start:end],
                site[start:end],
                decisions,
            )
            point = point_metric_statistics(prediction.cpu().numpy(), truth_numpy)
            metric["pas"][index, start:end] = point.pas
            metric["pdp"][index, start:end] = point.pdp
            metric["error"][index, start:end] = point.squared_error
            metric["energy"][index, start:end] = point.truth_energy
            if name == "identity":
                reference = torch.as_tensor(
                    np.array(v4_reference[start:end], dtype=np.complex64, copy=True),
                    device=device,
                )
                difference = prediction - reference
                identity_difference_energy += float(
                    difference.abs().square().sum(dtype=torch.float64).item()
                )
                identity_reference_energy += float(
                    reference.abs().square().sum(dtype=torch.float64).item()
                )
                identity_max_absolute = max(
                    identity_max_absolute, float(difference.abs().max().item())
                )
        if end % 20 == 0 or end == len(query_ids):
            print(f"preprojection wall phase {end}/{len(query_ids)}", flush=True)

    diagnostics = {
        "identity_relative_squared_difference": float(
            identity_difference_energy / max(identity_reference_energy, 1e-30)
        ),
        "identity_max_absolute_difference": identity_max_absolute,
        "base_component_explained_fraction": float(
            1.0 - residual_energy / max(base_energy, 1e-30)
        ),
        "active_path_fraction": float(np.mean(phase_mask)),
        "active_point_fraction": float(np.mean(np.any(phase_mask, axis=1))),
    }
    np.savez_compressed(
        METRIC_CACHE,
        **metric,
        variant_names=np.asarray(VARIANT_NAMES),
        fold_id=fold_id,
        site=site,
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    return summarize(metric, fold_id, site, config), diagnostics


def summarize(
    metric: dict[str, np.ndarray],
    fold_id: np.ndarray,
    site: np.ndarray,
    config: Config,
) -> dict[str, object]:
    candidates = candidate_statistics(VARIANT_NAMES, metric, fold_id)
    by_name = {str(row["name"]): row for row in candidates}
    baseline_score = float(by_name["identity"]["score"])
    for row in candidates:
        row["delta_vs_identity"] = float(row["score"] - baseline_score)
        row["fold_delta_vs_identity"] = [
            float(current["score"] - old["score"])
            for current, old in zip(row["folds"], by_name["identity"]["folds"])
        ]
        row["positive_fold_count"] = int(
            np.sum(np.asarray(row["fold_delta_vs_identity"]) > 0.0)
        )
    nested = nested_select_real_action(metric, fold_id, site)
    controls = (
        float(by_name["rotated_geometry_control"]["score"]),
        float(by_name["wrong_wavenumber_control"]["score"]),
    )
    correct_margin = float(by_name["correct_geometry"]["score"] - max(controls))
    deployment_ready = bool(
        float(nested["score"]) >= config.deployment_score_threshold
        and int(nested["positive_fold_count"]) >= config.deployment_positive_folds
        and correct_margin > 0.0
    )
    return {
        "baseline": by_name["identity"],
        "candidates": sorted(
            candidates, key=lambda row: float(row["score"]), reverse=True
        ),
        "nested_selection": nested,
        "correct_margin_over_best_control": correct_margin,
        "controls_selectable": False,
        "deployment_ready": deployment_ready,
        "deployment_rule": {
            "minimum_score": config.deployment_score_threshold,
            "minimum_positive_folds": config.deployment_positive_folds,
            "correct_must_beat_both_controls": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in np.load(PHASE_CACHE).items()}
    evaluation, diagnostics = evaluate(
        arrays,
        config,
        torch.device(args.device),
        args.batch_size,
        args.rebuild,
    )
    report = {
        "experiment": VERSION,
        "status": "candidate_ready" if evaluation["deployment_ready"] else "analysis_only",
        "primary_hypothesis": (
            "Deployable wall-geometry carrier phase must be inserted into the sparse "
            "pre-projection carrier so V4 can rebuild its validated PAS/PDP amplitudes "
            "around the improved phase instead of editing mixed V4 output modes."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "config": asdict(config),
        "frozen_v4_operator": True,
        "evaluation": evaluation,
        "diagnostics": diagnostics,
        "outputs": {"metrics": PATHS.relative(METRIC_CACHE)},
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
