#!/usr/bin/env python3
"""Evaluate calibrated Sionna PAS/PDP targets through the frozen V4 carrier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import time

import numpy as np
import torch

from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    fixed_blend,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
    truth_views,
)
from geometry_features import GEOMETRY_CACHE
from prepare_sionna_calibration_dataset import ARTIFACT_DIR, TRUTH_PAS, TRUTH_PDP
from project_paths import PATHS
from round2.data import load_positions
from run_rbf_spectral_field import OOF_META, statistics


VERSION = "Round2-E40-calibrated-Sionna-projection"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_sionna_official_calibration"
REPORT = ANALYSIS_DIR / "calibrated_projection.json"
METRICS = ARTIFACT_DIR / "calibrated_projection_metrics.npz"
V4_OOF = (
    PATHS.artifacts
    / "round2"
    / "v5_static_link_phase_calibration"
    / "v4_nested_oof_channel.npy"
)
PAS_TARGET = ARTIFACT_DIR / "oof_pas_mean_f16.npy"
PDP_TARGET = ARTIFACT_DIR / "oof_pdp_mean_f16.npy"
TINY = 1e-30


@dataclass(frozen=True)
class Candidate:
    name: str
    alpha_pas: float = 0.0
    alpha_pdp: float = 0.0
    order: str = "pas_pdp"
    cycles: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def candidates() -> list[Candidate]:
    result = [Candidate("identity")]
    pas_alphas = (0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
    pdp_alphas = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00)
    result.extend(
        Candidate(f"pas_a{alpha:.3f}", alpha_pas=alpha) for alpha in pas_alphas
    )
    result.extend(
        Candidate(f"pdp_a{alpha:.3f}", alpha_pdp=alpha) for alpha in pdp_alphas
    )
    for alpha_pas in pas_alphas:
        for alpha_pdp in pdp_alphas:
            for order in ("pas_pdp", "pdp_pas"):
                for cycles in (4, 16):
                    result.append(
                        Candidate(
                            name=(
                                f"dual_ap{alpha_pas:.3f}_ad{alpha_pdp:.3f}_"
                                f"{order}_c{cycles}"
                            ),
                            alpha_pas=alpha_pas,
                            alpha_pdp=alpha_pdp,
                            order=order,
                            cycles=cycles,
                        )
                    )
    return result


def apply_candidate(
    base: torch.Tensor,
    pas_target: torch.Tensor,
    pdp_target: torch.Tensor,
    spec: Candidate,
) -> torch.Tensor:
    if spec.name == "identity":
        return base
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    desired_pas = fixed_blend(
        pas_distribution(base), pas_target, spec.alpha_pas, (1, 2, 3)
    )
    desired_pdp = fixed_blend(
        pdp_distribution(base), pdp_target, spec.alpha_pdp, (3,)
    )
    candidate = base
    domains = []
    if spec.alpha_pas > 0.0:
        domains.append("pas")
    if spec.alpha_pdp > 0.0:
        domains.append("pdp")
    if len(domains) > 1:
        domains = spec.order.split("_")
    for _ in range(spec.cycles):
        for domain in domains:
            if domain == "pas":
                candidate = exact_pas_projection(candidate, desired_pas)
            elif domain == "pdp":
                candidate = exact_pdp_projection(candidate, desired_pdp)
            else:
                raise ValueError(domain)
        candidate = preserve_point_power(candidate, reference_power)
    return candidate


def point_metrics(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shaped = prediction.reshape(len(prediction), 2, 16, 8, 4, 192)
    prediction_pas = torch.fft.fft2(shaped, dim=(2, 3)).abs().square().float()
    prediction_pdp = torch.fft.ifft(prediction, dim=-1).abs().square().float()

    pas_num = (prediction_pas * truth_pas).sum(dim=(1, 2, 3), dtype=torch.float64)
    pas_den = torch.sqrt(
        prediction_pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
        * truth_pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
    )
    pas_den = torch.where(pas_den == 0.0, torch.full_like(pas_den, 1e-12), pas_den)
    pas = (pas_num / pas_den).mean(dim=(1, 2), dtype=torch.float64)

    pdp_num = (prediction_pdp * truth_pdp).sum(dim=3, dtype=torch.float64)
    pdp_den = torch.sqrt(
        prediction_pdp.square().sum(dim=3, dtype=torch.float64)
        * truth_pdp.square().sum(dim=3, dtype=torch.float64)
    )
    pdp_den = torch.where(pdp_den == 0.0, torch.full_like(pdp_den, 1e-12), pdp_den)
    pdp = (pdp_num / pdp_den).mean(dim=(1, 2), dtype=torch.float64)
    error = (prediction - truth).abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    energy = truth.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    return tuple(value.cpu().numpy() for value in (pas, pdp, error, energy))


def generate_metrics(device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    specs = candidates()
    count = len(query_ids)
    output = {
        "pas": np.empty((len(specs), count), dtype=np.float64),
        "pdp": np.empty((len(specs), count), dtype=np.float64),
        "error": np.empty((len(specs), count), dtype=np.float64),
        "energy": np.empty((len(specs), count), dtype=np.float64),
    }
    base_all = np.load(V4_OOF, mmap_mode="r")
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    pas_all = np.load(PAS_TARGET, mmap_mode="r")
    pdp_all = np.load(PDP_TARGET, mmap_mode="r")
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        base = torch.as_tensor(
            np.array(base_all[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        pas_target = torch.as_tensor(
            np.array(pas_all[start:end], dtype=np.float32, copy=True), device=device
        )
        pdp_target = torch.as_tensor(
            np.array(pdp_all[start:end], dtype=np.float32, copy=True), device=device
        )
        with torch.inference_mode():
            truth_pas, truth_pdp = truth_views(truth)
            for index, spec in enumerate(specs):
                prediction = apply_candidate(base, pas_target, pdp_target, spec)
                values = point_metrics(prediction, truth, truth_pas, truth_pdp)
                for key, value in zip(("pas", "pdp", "error", "energy"), values):
                    output[key][index, start:end] = value
        print(f"calibrated projection {end}/{count}", flush=True)
    output.update(
        {
            "candidate_names": np.asarray([spec.name for spec in specs]),
            "query_ids": query_ids,
            "fold_id": fold_id,
            "site": site,
        }
    )
    return output


def candidate_result(
    metrics: dict[str, np.ndarray], index: int, mask: np.ndarray | None = None
) -> dict[str, float | int]:
    return statistics(
        metrics["pas"][index],
        metrics["pdp"][index],
        metrics["error"][index],
        metrics["energy"][index],
        mask,
    )


def selected_result(
    metrics: dict[str, np.ndarray], selected: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float | int]:
    point = np.arange(len(selected))
    if mask is None:
        mask = np.ones(len(selected), dtype=bool)
    return statistics(
        metrics["pas"][selected[mask], point[mask]],
        metrics["pdp"][selected[mask], point[mask]],
        metrics["error"][selected[mask], point[mask]],
        metrics["energy"][selected[mask], point[mask]],
    )


def nested_policy(metrics: dict[str, np.ndarray]) -> tuple[np.ndarray, list[dict[str, object]]]:
    names = metrics["candidate_names"]
    fold_id = metrics["fold_id"]
    site = metrics["site"]
    selected = np.zeros(len(fold_id), dtype=np.int64)
    decisions = []
    for held_fold in np.unique(fold_id):
        for site_value in np.unique(site):
            fit = (fold_id != held_fold) & (site == site_value)
            held = (fold_id == held_fold) & (site == site_value)
            rows = [candidate_result(metrics, index, fit) for index in range(len(names))]
            best = int(np.argmax([row["score"] for row in rows]))
            selected[held] = best
            decisions.append(
                {
                    "fold": int(held_fold),
                    "site": int(site_value),
                    "candidate": str(names[best]),
                    "fit": rows[best],
                    "held": candidate_result(metrics, best, held),
                }
            )
    return selected, decisions


def oracle_policy(metrics: dict[str, np.ndarray]) -> np.ndarray:
    score = (
        0.4 * metrics["pas"]
        + 0.4 * metrics["pdp"]
        + 0.2 / (1.0 + metrics["error"] / np.maximum(metrics["energy"], TINY))
    )
    return np.argmax(score, axis=0).astype(np.int64)


def physical_groups(query_ids: np.ndarray) -> dict[str, np.ndarray]:
    geometry = np.load(GEOMETRY_CACHE, allow_pickle=False)
    names = geometry["names"].astype(str).tolist()
    values = geometry["train"][query_ids]
    center = values[:, names.index("center_hit_count")]
    fan_clear = values[:, names.index("fan_clear_fraction")]
    return {
        "clear": (center == 0.0) & (fan_clear >= 0.8),
        "light": (center == 0.0) & (fan_clear < 0.8),
        "medium": (center > 0.0) & (center < 2.0),
        "heavy": center >= 2.0,
    }


def summarize_policy(
    metrics: dict[str, np.ndarray], selected: np.ndarray, identity_index: int
) -> dict[str, object]:
    fold_id = metrics["fold_id"]
    site = metrics["site"]
    groups = physical_groups(metrics["query_ids"])
    result = {
        "overall": selected_result(metrics, selected),
        "folds": [],
        "holes": [],
        "physical_groups": {},
    }
    for fold in np.unique(fold_id):
        mask = fold_id == fold
        current = selected_result(metrics, selected, mask)
        base = candidate_result(metrics, identity_index, mask)
        result["folds"].append(
            {
                "fold": int(fold),
                "result": current,
                "delta_score": float(current["score"] - base["score"]),
            }
        )
        for site_value in np.unique(site):
            hole = mask & (site == site_value)
            current_hole = selected_result(metrics, selected, hole)
            base_hole = candidate_result(metrics, identity_index, hole)
            result["holes"].append(
                {
                    "fold": int(fold),
                    "site": int(site_value),
                    "result": current_hole,
                    "delta_score": float(current_hole["score"] - base_hole["score"]),
                }
            )
    for name, mask in groups.items():
        if not np.any(mask):
            continue
        current = selected_result(metrics, selected, mask)
        base = candidate_result(metrics, identity_index, mask)
        result["physical_groups"][name] = {
            "result": current,
            "delta_score": float(current["score"] - base["score"]),
        }
    return result


def direct_target_scores(query_ids: np.ndarray) -> dict[str, float]:
    predicted_pas = np.asarray(np.load(PAS_TARGET, mmap_mode="r"), dtype=np.float32)
    predicted_pdp = np.asarray(np.load(PDP_TARGET, mmap_mode="r"), dtype=np.float32)
    truth_pas = np.asarray(np.load(TRUTH_PAS, mmap_mode="r")[query_ids], dtype=np.float32)
    truth_pdp = np.asarray(np.load(TRUTH_PDP, mmap_mode="r")[query_ids], dtype=np.float32)

    pas_num = np.sum(predicted_pas * truth_pas, axis=(1, 2, 3), dtype=np.float64)
    pas_den = np.sqrt(
        np.sum(predicted_pas**2, axis=(1, 2, 3), dtype=np.float64)
        * np.sum(truth_pas**2, axis=(1, 2, 3), dtype=np.float64)
    )
    pas = np.divide(pas_num, np.where(pas_den == 0.0, 1e-12, pas_den)).mean(
        axis=(1, 2)
    )
    pdp_num = np.sum(predicted_pdp * truth_pdp, axis=3, dtype=np.float64)
    pdp_den = np.sqrt(
        np.sum(predicted_pdp**2, axis=3, dtype=np.float64)
        * np.sum(truth_pdp**2, axis=3, dtype=np.float64)
    )
    pdp = np.divide(pdp_num, np.where(pdp_den == 0.0, 1e-12, pdp_den)).mean(
        axis=(1, 2)
    )
    return {"pas": float(np.mean(pas)), "pdp": float(np.mean(pdp))}


def main() -> None:
    args = parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if args.reuse:
        metrics = {key: value for key, value in np.load(METRICS).items()}
    else:
        metrics = generate_metrics(torch.device(args.device), args.batch_size)
        np.savez_compressed(METRICS, **metrics)
    rows = [
        {"candidate": str(name), **candidate_result(metrics, index)}
        for index, name in enumerate(metrics["candidate_names"])
    ]
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    identity_index = int(np.flatnonzero(metrics["candidate_names"] == "identity")[0])
    identity = candidate_result(metrics, identity_index)
    fixed_best_index = int(
        np.flatnonzero(metrics["candidate_names"] == rows[0]["candidate"])[0]
    )
    fixed_selected = np.full(len(metrics["query_ids"]), fixed_best_index, dtype=np.int64)
    nested_selected, decisions = nested_policy(metrics)
    oracle_selected = oracle_policy(metrics)
    report = {
        "experiment": VERSION,
        "status": "complete",
        "input_policy": {
            "base": PATHS.relative(V4_OOF),
            "pas_target": PATHS.relative(PAS_TARGET),
            "pdp_target": PATHS.relative(PDP_TARGET),
            "point_power_preserved": True,
            "absolute_sionna_phase_used": False,
        },
        "candidate_count": int(len(metrics["candidate_names"])),
        "direct_target_scores": direct_target_scores(metrics["query_ids"]),
        "identity": identity,
        "fixed_best": rows[0],
        "fixed_best_policy": summarize_policy(metrics, fixed_selected, identity_index),
        "nested": summarize_policy(metrics, nested_selected, identity_index),
        "nested_decisions": decisions,
        "pointwise_oracle": summarize_policy(metrics, oracle_selected, identity_index),
        "oracle_changed_fraction": float(np.mean(oracle_selected != identity_index)),
        "top_candidates": rows[:30],
        "metrics": PATHS.relative(METRICS),
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
