#!/usr/bin/env python3
"""Evaluate the complete-score ceiling of locked-PDP soft angle transport."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

import analyze_e100_locked_energy_angle_correction as e100
import analyze_e101_deduplicated_angle_residual as e101
import analyze_e102_locked_pdp_candidate_reassignment_oracle as e102
import analyze_e86_surface_endpoint_v11_blend as e86
import optimize_e98_global_array_angle_mapping as e98
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E103-soft-transport-complete-ceiling"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "soft_transport_complete_ceiling"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_soft_transport_complete_ceiling"
REPORT = ANALYSIS_DIR / "soft_transport_complete_ceiling.json"
CEILING = ARTIFACT_DIR / "soft_transport_complete_ceiling.npz"


def complete_metrics(
    name: str,
    soft_pas: np.ndarray,
    endpoint_pdp: torch.Tensor,
    base: torch.Tensor,
    base_pas: torch.Tensor,
    base_pdp: torch.Tensor,
    truth: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    endpoint_pas = (
        torch.as_tensor(soft_pas, device=device)[..., None]
        .expand(-1, -1, -1, -1, -1, 192)
    )
    spec = e86.Candidate(
        endpoint=name,
        mode="sqrt",
        pas_beta=0.20,
        pdp_beta=0.20,
        order="pas_pdp",
        cycles=4,
    )
    with torch.inference_mode():
        prediction = e86.make_candidate(
            base,
            base_pas,
            base_pdp,
            endpoint_pas,
            endpoint_pdp,
            spec,
        )
        return point_metrics(prediction, truth, truth_pas, truth_pdp)


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    required = (
        e100.OLD_ENDPOINT,
        e100.MAPPING_PARAMETERS,
        e100.V12_CHANNEL,
        e85.BASE_COORDINATE,
        e86.V11_CHANNEL,
        e86.TRAIN_CHANNEL,
    )
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    store = e85.DataStore()
    query_ids = store.strict_ids
    fold_id = store.strict_fold
    endpoint = np.load(e100.OLD_ENDPOINT, allow_pickle=False)
    mass = endpoint["mass"].astype(np.float32)
    port = endpoint["port"].astype(np.float32)
    old_coordinate = endpoint["coordinate"].astype(np.float32)
    base_coordinate = np.asarray(
        np.load(e85.BASE_COORDINATE, mmap_mode="r")[query_ids], dtype=np.float32
    )
    mapping = np.load(e100.MAPPING_PARAMETERS, allow_pickle=False)
    mapping_data = e98.MappingData(device)
    corrected_base, _ = e101.build_mapping_delta(
        query_ids,
        fold_id,
        base_coordinate,
        mapping_data,
        mapping["refit_latent"].astype(np.float32),
        device,
    )
    valid = mapping_data.candidate_valid[
        torch.as_tensor(query_ids, device=device)
    ].cpu().numpy()
    target_pas = np.asarray(store.target_pas[query_ids], dtype=np.float32)
    cases = {
        "old_final": (old_coordinate, valid),
        "union": (
            np.concatenate((old_coordinate, corrected_base), axis=1),
            np.concatenate((valid, valid), axis=1),
        ),
    }
    result = {}
    for name, (candidate_coordinate, candidate_valid) in cases.items():
        result[name] = e102.optimize_case(
            name,
            mass,
            port,
            old_coordinate,
            candidate_coordinate,
            candidate_valid,
            target_pas,
            device,
            batch_size=50,
            steps=200,
            learning_rate=0.12,
        )

    old_output = {
        "mass": torch.as_tensor(mass, device=device),
        "coordinate": torch.as_tensor(old_coordinate, device=device),
        "port": torch.as_tensor(port, device=device),
    }
    with torch.inference_mode():
        _, endpoint_pdp_small = e85.render_spectra(old_output)
        endpoint_pdp = (
            endpoint_pdp_small[:, :, None, None, :, :]
            .expand(-1, -1, 16, 8, -1, -1)
            .reshape(-1, 256, 4, 192)
        )

    base = torch.as_tensor(
        np.array(np.load(e86.V11_CHANNEL, mmap_mode="r"), copy=True), device=device
    )
    truth = torch.as_tensor(
        np.asarray(
            np.load(e86.TRAIN_CHANNEL, mmap_mode="r")[query_ids],
            dtype=np.complex64,
        ),
        device=device,
    )
    v12 = torch.as_tensor(
        np.array(np.load(e100.V12_CHANNEL, mmap_mode="r"), copy=True), device=device
    )
    with torch.inference_mode():
        base_pas = pas_distribution(base)
        base_pdp = pdp_distribution(base)
        truth_pas = pas_distribution(truth)
        truth_pdp = pdp_distribution(truth)
        v12_metrics = point_metrics(v12, truth, truth_pas, truth_pdp)

    old_metrics = complete_metrics(
        "old_soft_oracle",
        np.asarray(result["old_final"]["soft_pas"], dtype=np.float32),
        endpoint_pdp,
        base,
        base_pas,
        base_pdp,
        truth,
        truth_pas,
        truth_pdp,
        device,
    )
    union_metrics = complete_metrics(
        "union_soft_oracle",
        np.asarray(result["union"]["soft_pas"], dtype=np.float32),
        endpoint_pdp,
        base,
        base_pas,
        base_pdp,
        truth,
        truth_pas,
        truth_pdp,
        device,
    )
    old_vs_v12 = e100.metric_comparison(old_metrics, v12_metrics, fold_id)
    union_vs_v12 = e100.metric_comparison(union_metrics, v12_metrics, fold_id)
    union_vs_old = e100.metric_comparison(union_metrics, old_metrics, fold_id)

    old_soft = np.asarray(result["old_final"]["soft_score"], dtype=np.float64)
    union_soft = np.asarray(result["union"]["soft_score"], dtype=np.float64)
    score_reproduction = {
        "old_soft_pas": float(np.mean(old_soft)),
        "union_soft_pas": float(np.mean(union_soft)),
        "union_gain": float(np.mean(union_soft - old_soft)),
    }
    decision = {
        "minimum_complete_gain_union_vs_old": 0.001,
        "minimum_positive_folds": 4,
        "complete_gain_union_vs_old": union_vs_old["delta"],
        "positive_folds": union_vs_old["positive_folds"],
        "worst_fold_delta": union_vs_old["worst_fold_delta"],
    }
    decision["train_soft_transport"] = bool(
        decision["complete_gain_union_vs_old"]
        >= decision["minimum_complete_gain_union_vs_old"]
        and decision["positive_folds"] >= decision["minimum_positive_folds"]
    )

    np.savez_compressed(
        CEILING,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        old_soft_pas=np.asarray(result["old_final"]["soft_pas"], dtype=np.float16),
        union_soft_pas=np.asarray(result["union"]["soft_pas"], dtype=np.float16),
        old_point_pas=old_soft.astype(np.float32),
        union_point_pas=union_soft.astype(np.float32),
    )
    report = {
        "experiment": VERSION,
        "status": "strict_oracle_complete",
        "primary_hypothesis": (
            "The E102 soft union PAS advantage remains material after the frozen "
            "V12 complete-channel projection."
        ),
        "validation": (
            "Canonical 500 held truths used only for oracle transport. The comparison "
            "isolates old-candidate soft transport versus old+E98 soft transport."
        ),
        "optimization": {
            "batch_size": 50,
            "steps": 200,
            "learning_rate": 0.12,
            "initialization": "nearest candidate with logit 4.0",
            "v12_policy": "sqrt PAS/PDP 0.20/0.20, PAS->PDP, 4 cycles",
            "mass_port_delay_pdp_power_locked": True,
            "score_reproduction": score_reproduction,
        },
        "complete_score": {
            "old_soft_vs_v12": old_vs_v12,
            "union_soft_vs_v12": union_vs_v12,
            "union_soft_vs_old_soft": union_vs_old,
        },
        "decision": {**decision, "generate_v13": False},
        "outputs": {
            "ceiling": PATHS.relative(CEILING),
            "soft_pas_shape": list(
                np.asarray(result["union"]["soft_pas"]).shape
            ),
            "finite": bool(
                np.isfinite(result["old_final"]["soft_pas"]).all()
                and np.isfinite(result["union"]["soft_pas"]).all()
            ),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
