#!/usr/bin/env python3
"""Measure the locked-PDP PAS ceiling of E98 candidate reassignment."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

import analyze_e100_locked_energy_angle_correction as e100
import analyze_e101_deduplicated_angle_residual as e101
import analyze_e86_surface_endpoint_v11_blend as e86
import optimize_e98_global_array_angle_mapping as e98
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E102-locked-pdp-candidate-reassignment-oracle"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "locked_pdp_candidate_reassignment_oracle"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_locked_pdp_candidate_reassignment_oracle"
)
REPORT = ANALYSIS_DIR / "locked_pdp_candidate_reassignment_oracle.json"
ORACLE = ARTIFACT_DIR / "candidate_reassignment_oracle.npz"
TINY = 1e-30


def periodic_delta(value: torch.Tensor, period: float) -> torch.Tensor:
    return torch.remainder(value + period / 2.0, period) - period / 2.0


def summarize(values: np.ndarray, fold_id: np.ndarray) -> dict[str, object]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "fold": [
            {
                "fold": outer,
                "count": int(np.sum(fold_id == outer)),
                "mean": float(np.mean(values[fold_id == outer])),
            }
            for outer in range(5)
        ],
    }


def compare(
    selected: np.ndarray, baseline: np.ndarray, fold_id: np.ndarray
) -> dict[str, object]:
    delta = selected - baseline
    folds = []
    for outer in range(5):
        mask = fold_id == outer
        folds.append(
            {
                "fold": outer,
                "selected": float(np.mean(selected[mask])),
                "baseline": float(np.mean(baseline[mask])),
                "delta": float(np.mean(delta[mask])),
            }
        )
    return {
        "selected": float(np.mean(selected)),
        "baseline": float(np.mean(baseline)),
        "delta": float(np.mean(delta)),
        "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
        "worst_fold_delta": float(min(row["delta"] for row in folds)),
        "folds": folds,
    }


def candidate_kernels(coordinate: torch.Tensor) -> torch.Tensor:
    horizontal, vertical, _ = e85.spectral_kernels(coordinate)
    return horizontal[..., :, None] * vertical[..., None, :]


def assignment_pas(
    energy: torch.Tensor,
    assignment: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    assigned_energy = torch.einsum("bipu,bij->bjpu", energy, assignment)
    pas = torch.einsum("bjpu,bjhv->bphvu", assigned_energy, kernel)
    return pas / pas.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)


def pas_score(pas: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return e85.cosine_sufficient_score(pas, target, (1, 2, 3), (1,))


def optimize_case(
    name: str,
    mass: np.ndarray,
    port: np.ndarray,
    old_coordinate: np.ndarray,
    candidate_coordinate: np.ndarray,
    candidate_valid: np.ndarray,
    target_pas: np.ndarray,
    device: torch.device,
    batch_size: int = 50,
    steps: int = 200,
    learning_rate: float = 0.12,
) -> dict[str, object]:
    count, paths = mass.shape
    soft_score = np.zeros(count, dtype=np.float64)
    hard_score = np.zeros(count, dtype=np.float64)
    soft_pas = np.zeros((count, 2, 16, 8, 4), dtype=np.float32)
    hard_index = np.zeros((count, paths), dtype=np.int16)
    histories = []
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        local_mass = torch.as_tensor(mass[start:end], device=device)
        local_port = torch.as_tensor(port[start:end], device=device)
        energy = local_mass[..., None, None] * local_port
        old = torch.as_tensor(old_coordinate[start:end], device=device)
        candidate = torch.as_tensor(candidate_coordinate[start:end], device=device)
        valid = torch.as_tensor(candidate_valid[start:end], device=device)
        target = torch.as_tensor(target_pas[start:end], device=device)
        if not torch.all(valid.any(dim=1)):
            raise ValueError(f"case {name} has a point without a valid candidate")
        kernel = candidate_kernels(candidate)

        dh = periodic_delta(
            candidate[:, None, :, 0] - old[:, :, None, 0], 16.0
        )
        dv = periodic_delta(
            candidate[:, None, :, 1] - old[:, :, None, 1], 8.0
        )
        cost = (dh / 1.5).square() + dv.square()
        cost = cost.masked_fill(~valid[:, None, :], 1e9)
        nearest = cost.argmin(dim=2)
        logits = torch.zeros(
            len(local_mass), paths, candidate.shape[1], device=device
        )
        logits.scatter_(2, nearest[..., None], 4.0)
        logits = torch.nn.Parameter(logits)
        optimizer = torch.optim.Adam([logits], lr=learning_rate)
        history = []
        for step in range(steps + 1):
            masked_logits = logits.masked_fill(~valid[:, None, :], -1e4)
            assignment = torch.softmax(masked_logits, dim=2)
            prediction = assignment_pas(energy, assignment, kernel)
            score = pas_score(prediction, target)
            if step in (0, 25, 50, 100, 150, steps):
                history.append(
                    {"step": step, "mean_soft_pas": float(score.mean().item())}
                )
            if step == steps:
                break
            loss = -score.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if logits.grad is None or not torch.isfinite(logits.grad).all():
                raise FloatingPointError(
                    f"non-finite assignment gradient case={name} step={step}"
                )
            optimizer.step()

        with torch.inference_mode():
            masked_logits = logits.masked_fill(~valid[:, None, :], -1e4)
            assignment = torch.softmax(masked_logits, dim=2)
            soft_prediction = assignment_pas(energy, assignment, kernel)
            local_soft = pas_score(soft_prediction, target)
            local_index = assignment.argmax(dim=2)
            hard_assignment = torch.nn.functional.one_hot(
                local_index, num_classes=candidate.shape[1]
            ).to(torch.float32)
            hard_prediction = assignment_pas(energy, hard_assignment, kernel)
            local_hard = pas_score(hard_prediction, target)
        soft_score[start:end] = local_soft.cpu().numpy()
        hard_score[start:end] = local_hard.cpu().numpy()
        soft_pas[start:end] = soft_prediction.cpu().numpy().astype(np.float32)
        hard_index[start:end] = local_index.cpu().numpy().astype(np.int16)
        histories.append(
            {
                "start": start,
                "end": end,
                "history": history,
                "final_soft": float(np.mean(soft_score[start:end])),
                "final_hard": float(np.mean(hard_score[start:end])),
            }
        )
        print(
            f"E102 case={name} rows={end}/{count} "
            f"soft={np.mean(soft_score[start:end]):.6f} "
            f"hard={np.mean(hard_score[start:end]):.6f}",
            flush=True,
        )
    return {
        "soft_score": soft_score,
        "hard_score": hard_score,
        "soft_pas": soft_pas,
        "hard_index": hard_index,
        "histories": histories,
    }


def hard_coordinates(
    old_coordinate: np.ndarray,
    candidate_coordinate: np.ndarray,
    hard_index: np.ndarray,
) -> np.ndarray:
    selected = np.take_along_axis(
        candidate_coordinate,
        hard_index[..., None].astype(np.int64),
        axis=1,
    )
    coordinate = np.array(old_coordinate, copy=True)
    coordinate[..., :2] = selected[..., :2]
    return coordinate


@torch.inference_mode()
def render_endpoint(
    mass: np.ndarray,
    coordinate: np.ndarray,
    port: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = {
        "mass": torch.as_tensor(mass, device=device),
        "coordinate": torch.as_tensor(coordinate, device=device),
        "port": torch.as_tensor(port, device=device),
    }
    pas_small, pdp_small = e85.render_spectra(output)
    pas = (
        pas_small[..., None]
        .expand(-1, -1, -1, -1, -1, 192)
    )
    pdp = (
        pdp_small[:, :, None, None, :, :]
        .expand(-1, -1, 16, 8, -1, -1)
        .reshape(-1, 256, 4, 192)
    )
    return pas, pdp


def evaluate_complete_oracle(
    name: str,
    mass: np.ndarray,
    coordinate: np.ndarray,
    port: np.ndarray,
    base: torch.Tensor,
    base_pas: torch.Tensor,
    base_pdp: torch.Tensor,
    truth: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
    v12_metrics: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    locked_endpoint_pdp: torch.Tensor,
    fold_id: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, object], float]:
    endpoint_pas, endpoint_pdp = render_endpoint(mass, coordinate, port, device)
    pdp_difference = float(
        torch.max(torch.abs(endpoint_pdp - locked_endpoint_pdp)).item()
    )
    spec = e86.Candidate(
        endpoint=name,
        mode="sqrt",
        pas_beta=0.20,
        pdp_beta=0.20,
        order="pas_pdp",
        cycles=4,
    )
    prediction = e86.make_candidate(
        base, base_pas, base_pdp, endpoint_pas, endpoint_pdp, spec
    )
    selected_metrics = point_metrics(prediction, truth, truth_pas, truth_pdp)
    return e100.metric_comparison(selected_metrics, v12_metrics, fold_id), pdp_difference


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
    if not np.array_equal(endpoint["query_ids"].astype(np.int64), query_ids):
        raise ValueError("old endpoint query order does not match canonical 500 points")
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
    valid_count = valid.sum(axis=1)
    if np.any(valid_count == 0):
        raise ValueError("at least one strict point has no valid E98 candidate")
    target_pas = np.asarray(store.target_pas[query_ids], dtype=np.float32)

    cases = {
        "old_final": (old_coordinate, valid),
        "e98_corrected": (corrected_base, valid),
        "union": (
            np.concatenate((old_coordinate, corrected_base), axis=1),
            np.concatenate((valid, valid), axis=1),
        ),
    }
    results: dict[str, dict[str, object]] = {}
    for name, (candidate_coordinate, candidate_valid) in cases.items():
        results[name] = optimize_case(
            name,
            mass,
            port,
            old_coordinate,
            candidate_coordinate,
            candidate_valid,
            target_pas,
            device,
        )

    hard_coordinate: dict[str, np.ndarray] = {}
    for name, (candidate_coordinate, _) in cases.items():
        hard_coordinate[name] = hard_coordinates(
            old_coordinate,
            candidate_coordinate,
            results[name]["hard_index"],
        )

    base_np = np.load(e86.V11_CHANNEL, mmap_mode="r")
    truth_np = np.asarray(
        np.load(e86.TRAIN_CHANNEL, mmap_mode="r")[query_ids], dtype=np.complex64
    )
    base = torch.as_tensor(np.array(base_np, copy=True), device=device)
    truth = torch.as_tensor(truth_np, device=device)
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    v12 = torch.as_tensor(
        np.array(np.load(e100.V12_CHANNEL, mmap_mode="r"), copy=True), device=device
    )
    with torch.inference_mode():
        v12_metrics = point_metrics(v12, truth, truth_pas, truth_pdp)
        old_output = {
            "mass": torch.as_tensor(mass, device=device),
            "coordinate": torch.as_tensor(old_coordinate, device=device),
            "port": torch.as_tensor(port, device=device),
        }
        _, locked_pdp_small = e85.render_spectra(old_output)
        locked_endpoint_pdp = (
            locked_pdp_small[:, :, None, None, :, :]
            .expand(-1, -1, 16, 8, -1, -1)
            .reshape(-1, 256, 4, 192)
        )

    complete = {}
    for name in cases:
        comparison, pdp_difference = evaluate_complete_oracle(
            name,
            mass,
            hard_coordinate[name],
            port,
            base,
            base_pas,
            base_pdp,
            truth,
            truth_pas,
            truth_pdp,
            v12_metrics,
            locked_endpoint_pdp,
            fold_id,
            device,
        )
        complete[name] = {
            "versus_v12": comparison,
            "pdp_render_max_abs_difference": pdp_difference,
        }

    old_soft = np.asarray(results["old_final"]["soft_score"], dtype=np.float64)
    corrected_soft = np.asarray(
        results["e98_corrected"]["soft_score"], dtype=np.float64
    )
    union_soft = np.asarray(results["union"]["soft_score"], dtype=np.float64)
    old_hard = np.asarray(results["old_final"]["hard_score"], dtype=np.float64)
    corrected_hard = np.asarray(
        results["e98_corrected"]["hard_score"], dtype=np.float64
    )
    union_hard = np.asarray(results["union"]["hard_score"], dtype=np.float64)
    soft_union_gain = compare(union_soft, old_soft, fold_id)
    hard_union_gain = compare(union_hard, old_hard, fold_id)
    decision = {
        "minimum_soft_union_gain": 0.01,
        "minimum_hard_union_gain": 0.005,
        "minimum_positive_folds": 4,
        "soft_union_gain": soft_union_gain["delta"],
        "soft_positive_folds": soft_union_gain["positive_folds"],
        "hard_union_gain": hard_union_gain["delta"],
        "hard_positive_folds": hard_union_gain["positive_folds"],
    }
    decision["candidate_identity_has_material_ceiling"] = bool(
        decision["soft_union_gain"] >= decision["minimum_soft_union_gain"]
        and decision["soft_positive_folds"] >= decision["minimum_positive_folds"]
        and decision["hard_union_gain"] >= decision["minimum_hard_union_gain"]
        and decision["hard_positive_folds"] >= decision["minimum_positive_folds"]
    )

    np.savez_compressed(
        ORACLE,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        valid_count=valid_count.astype(np.int16),
        old_soft=old_soft.astype(np.float32),
        corrected_soft=corrected_soft.astype(np.float32),
        union_soft=union_soft.astype(np.float32),
        old_hard=old_hard.astype(np.float32),
        corrected_hard=corrected_hard.astype(np.float32),
        union_hard=union_hard.astype(np.float32),
        old_hard_index=np.asarray(results["old_final"]["hard_index"], dtype=np.int16),
        corrected_hard_index=np.asarray(
            results["e98_corrected"]["hard_index"], dtype=np.int16
        ),
        union_hard_index=np.asarray(results["union"]["hard_index"], dtype=np.int16),
    )
    report = {
        "experiment": VERSION,
        "status": "strict_oracle_complete",
        "primary_hypothesis": (
            "E98 improves PAS through candidate identity reassignment rather than "
            "same-index coordinate displacement."
        ),
        "validation": (
            "Canonical 500 holes with held truth used only as an explicit oracle. "
            "Mass, port allocation, delay, PDP, and total power remain locked."
        ),
        "optimization": {
            "cases": ["old_final", "e98_corrected", "union"],
            "batch_size": 50,
            "steps": 200,
            "learning_rate": 0.12,
            "initialization": "nearest candidate with logit 4.0",
            "candidate_valid_min_mean_max": [
                int(valid_count.min()),
                float(valid_count.mean()),
                int(valid_count.max()),
            ],
            "truth_usage": "oracle-only; never eligible for V13 packaging",
        },
        "pas_ceiling": {
            name: {
                "soft": summarize(
                    np.asarray(result["soft_score"], dtype=np.float64), fold_id
                ),
                "hard": summarize(
                    np.asarray(result["hard_score"], dtype=np.float64), fold_id
                ),
                "histories": result["histories"],
            }
            for name, result in results.items()
        },
        "incremental_information": {
            "corrected_soft_vs_old": compare(corrected_soft, old_soft, fold_id),
            "union_soft_vs_old": soft_union_gain,
            "corrected_hard_vs_old": compare(corrected_hard, old_hard, fold_id),
            "union_hard_vs_old": hard_union_gain,
        },
        "complete_hard_oracle": complete,
        "decision": {
            **decision,
            "train_deployable_selector": decision[
                "candidate_identity_has_material_ceiling"
            ],
            "generate_v13": False,
        },
        "outputs": {
            "oracle": PATHS.relative(ORACLE),
            "shape": [int(len(query_ids)), int(mass.shape[1])],
            "finite": bool(
                all(
                    np.isfinite(np.asarray(results[name]["soft_score"])).all()
                    and np.isfinite(np.asarray(results[name]["hard_score"])).all()
                    for name in results
                )
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
