#!/usr/bin/env python3
"""Evaluate a strict spatial residual field on the V9 PAS/PDP endpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from prepare_sionna_calibration_dataset import (
    ARTIFACT_DIR as E40_DIR,
    TRUTH_PAS,
    TRUTH_PDP,
)
from project_paths import PATHS
from round2.data import load_positions
from run_rbf_spectral_field import OOF_META, statistics
from run_v8_full_sionna_migration import reconstruct_v7_nested
from run_v9_joint_sionna_migration import Candidate, apply_candidate, point_metrics
from train_v9_joint_sionna_endpoints import output_path as joint_output_path


VERSION = "Round2-E59-V10-spatial-endpoint-residual-field"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v10_spatial_endpoint_residual_field"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v10_spatial_endpoint_residual_field"
METRICS = ARTIFACT_DIR / "spatial_endpoint_residual_metrics.npz"
REPORT = ANALYSIS_DIR / "spatial_endpoint_residual_field.json"
OLD_PAS = E40_DIR / "oof_pas_mean_f16.npy"
OLD_PDP = E40_DIR / "oof_pdp_mean_f16.npy"
JOINT_PAS = joint_output_path("pas", "oof")
JOINT_PDP = joint_output_path("pdp", "oof")
PAS_BETA = 0.75
PDP_BETA = 0.50
TINY = 1e-12


@dataclass(frozen=True)
class FieldSpec:
    neighbors: int
    distance_power: float
    residual_alpha: float

    @property
    def name(self) -> str:
        return (
            f"k{self.neighbors:02d}_p{int(round(10 * self.distance_power)):02d}_"
            f"a{int(round(100 * self.residual_alpha)):03d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--field-batch-size", type=int, default=4)
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def normalize_torch(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), 0.0)


def endpoint_blend(old: np.ndarray, joint: np.ndarray, beta: float, domain: str) -> np.ndarray:
    value = (1.0 - beta) * np.asarray(old, dtype=np.float32) + beta * np.asarray(
        joint, dtype=np.float32
    )
    axes = (1, 2, 3) if domain == "pas" else (3,)
    total = value.sum(axis=axes, keepdims=True, dtype=np.float64)
    return np.divide(
        value,
        np.maximum(total, TINY),
        out=np.zeros_like(value),
        where=total > 0.0,
    )


def build_donors(
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    neighbors: int,
    distance_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    donor_rows = np.empty((len(query_ids), neighbors), dtype=np.int32)
    weights = np.empty((len(query_ids), neighbors), dtype=np.float32)
    for fold in np.unique(fold_id):
        for site_value in np.unique(site[query_ids]):
            rows = np.flatnonzero(
                (fold_id == fold) & (site[query_ids] == site_value)
            )
            bank = np.flatnonzero(
                (fold_id != fold) & (site[query_ids] == site_value)
            )
            if not len(rows):
                continue
            tree = cKDTree(positions[query_ids[bank], :2])
            distance, local = tree.query(
                positions[query_ids[rows], :2], k=neighbors, workers=-1
            )
            distance = np.asarray(distance, dtype=np.float64).reshape(len(rows), -1)
            local = np.asarray(local, dtype=np.int64).reshape(len(rows), -1)
            selected = bank[local]
            local_weight = np.power(distance + 0.5, -distance_power)
            local_weight /= np.maximum(local_weight.sum(axis=1, keepdims=True), TINY)
            donor_rows[rows] = selected
            weights[rows] = local_weight.astype(np.float32)
    return donor_rows, weights


def weighted_field(
    residual: torch.Tensor,
    donors: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
) -> torch.Tensor:
    result = torch.empty_like(residual)
    device = residual.device
    for start in range(0, len(donors), batch_size):
        end = min(start + batch_size, len(donors))
        selected = torch.as_tensor(donors[start:end], device=device, dtype=torch.long)
        local_weight = torch.as_tensor(
            weights[start:end], device=device, dtype=torch.float32
        )
        gathered = residual[selected].float()
        shape = (end - start, gathered.shape[1]) + (1,) * (gathered.ndim - 2)
        result[start:end] = (
            gathered * local_weight.reshape(shape)
        ).sum(dim=1).to(result.dtype)
    return result


def corrected_endpoint(
    endpoint: torch.Tensor,
    field: torch.Tensor,
    alpha: float,
    domain: str,
) -> torch.Tensor:
    value = torch.clamp(torch.sqrt(endpoint.float().clamp_min(0.0)) + alpha * field.float(), min=0.0).square()
    dims = (1, 2, 3) if domain == "pas" else (3,)
    return normalize_torch(value, dims)


def endpoint_cosine(prediction: torch.Tensor, truth: torch.Tensor, domain: str) -> float:
    axes = (1, 2, 3) if domain == "pas" else (3,)
    average = (1, 2)
    numerator = (prediction * truth).sum(dim=axes, dtype=torch.float64)
    denominator = torch.sqrt(
        prediction.square().sum(dim=axes, dtype=torch.float64)
        * truth.square().sum(dim=axes, dtype=torch.float64)
    )
    cosine = numerator / torch.where(denominator == 0.0, 1e-12, denominator)
    return float(
        cosine.mean(dim=average, dtype=torch.float64)
        .mean(dtype=torch.float64)
        .cpu()
    )


def field_specs() -> list[FieldSpec]:
    return [
        FieldSpec(neighbors, power, alpha)
        for neighbors in (8, 16, 32, 64)
        for power in (1.0, 2.0)
        for alpha in (0.0, 0.15, 0.30, 0.50, 0.75, 1.0)
    ]


def prepare_domain(
    domain: str,
    device: torch.device,
    field_batch_size: int,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    old_path = OLD_PAS if domain == "pas" else OLD_PDP
    joint_path = JOINT_PAS if domain == "pas" else JOINT_PDP
    truth_path = TRUTH_PAS if domain == "pas" else TRUTH_PDP
    beta = PAS_BETA if domain == "pas" else PDP_BETA
    endpoint_numpy = endpoint_blend(
        np.load(old_path, mmap_mode="r"),
        np.load(joint_path, mmap_mode="r"),
        beta,
        domain,
    )
    truth_numpy = np.asarray(
        np.load(truth_path, mmap_mode="r")[query_ids], dtype=np.float32
    )
    endpoint = torch.as_tensor(endpoint_numpy, device=device, dtype=torch.bfloat16)
    truth = torch.as_tensor(truth_numpy, device=device, dtype=torch.bfloat16)
    residual = (
        torch.sqrt(truth.float().clamp_min(0.0))
        - torch.sqrt(endpoint.float().clamp_min(0.0))
    ).to(torch.bfloat16)
    baseline = endpoint_cosine(endpoint.float(), truth.float(), domain)
    rows: list[dict[str, object]] = []
    cache: dict[tuple[int, float], torch.Tensor] = {}
    for neighbors in (8, 16, 32, 64):
        for power in (1.0, 2.0):
            donors, weights = build_donors(
                query_ids, fold_id, positions, site, neighbors, power
            )
            field = weighted_field(
                residual, donors, weights, field_batch_size
            )
            cache[(neighbors, power)] = field
            for alpha in (0.0, 0.15, 0.30, 0.50, 0.75, 1.0):
                prediction = corrected_endpoint(endpoint, field, alpha, domain)
                score = endpoint_cosine(prediction, truth.float(), domain)
                fold_scores = []
                for fold in range(5):
                    mask = torch.as_tensor(fold_id == fold, device=device)
                    fold_scores.append(
                        endpoint_cosine(prediction[mask], truth.float()[mask], domain)
                    )
                spec = FieldSpec(neighbors, power, alpha)
                rows.append(
                    {
                        "spec": asdict(spec),
                        "name": spec.name,
                        "cosine": score,
                        "delta": score - baseline,
                        "fold_cosine": fold_scores,
                    }
                )
            print(
                f"V10 endpoint domain={domain} k={neighbors} power={power}",
                flush=True,
            )
    rows.sort(key=lambda row: float(row["cosine"]), reverse=True)
    selected_rows = rows[:6]
    baseline_row = next(
        row
        for row in rows
        if row["spec"]["neighbors"] == 8
        and row["spec"]["distance_power"] == 1.0
        and row["spec"]["residual_alpha"] == 0.0
    )
    if all(row["name"] != baseline_row["name"] for row in selected_rows):
        selected_rows.append(baseline_row)
    outputs: dict[str, np.ndarray] = {}
    for row in selected_rows:
        spec = FieldSpec(**row["spec"])
        prediction = corrected_endpoint(
            endpoint, cache[(spec.neighbors, spec.distance_power)], spec.residual_alpha, domain
        )
        outputs[spec.name] = prediction.cpu().numpy().astype(np.float16)
    del endpoint, truth, residual, cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        {
            "baseline_cosine": baseline,
            "selected": selected_rows,
            "top20": rows[:20],
        },
        outputs,
    )


def evaluate_full_channel(
    pas_outputs: dict[str, np.ndarray],
    pdp_outputs: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    base = reconstruct_v7_nested(device, batch_size, True)
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth_pas_all = np.load(TRUTH_PAS, mmap_mode="r")
    truth_pdp_all = np.load(TRUTH_PDP, mmap_mode="r")
    combinations: list[tuple[str, str, float, float]] = []
    for pas_name in pas_outputs:
        for pdp_name in pdp_outputs:
            combinations.append((pas_name, pdp_name, 0.45, 0.35))
    baseline_pas = next(
        name for name in pas_outputs if name.endswith("a000")
    )
    baseline_pdp = next(
        name for name in pdp_outputs if name.endswith("a000")
    )
    for alpha_pas in (0.40, 0.45, 0.50, 0.55):
        for alpha_pdp in (0.30, 0.35, 0.40, 0.45):
            combinations.append(
                (
                    next(iter(pas_outputs)),
                    next(iter(pdp_outputs)),
                    alpha_pas,
                    alpha_pdp,
                )
            )
    combinations = list(dict.fromkeys(combinations))
    shape = (len(combinations), len(query_ids))
    metrics = {
        key: np.empty(shape, dtype=np.float64)
        for key in ("pas", "pdp", "error", "energy")
    }
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        truth_pas = torch.as_tensor(
            np.asarray(truth_pas_all[query_ids[start:end]], dtype=np.float32),
            device=device,
        )
        truth_pdp = torch.as_tensor(
            np.asarray(truth_pdp_all[query_ids[start:end]], dtype=np.float32),
            device=device,
        )
        for index, (pas_name, pdp_name, alpha_pas, alpha_pdp) in enumerate(combinations):
            pas = torch.as_tensor(
                np.asarray(pas_outputs[pas_name][start:end], dtype=np.float32),
                device=device,
            )
            pdp = torch.as_tensor(
                np.asarray(pdp_outputs[pdp_name][start:end], dtype=np.float32),
                device=device,
            )
            spec = Candidate(
                name="v10",
                alpha_pas=alpha_pas,
                alpha_pdp=alpha_pdp,
                cycles=4,
            )
            with torch.inference_mode():
                prediction = apply_candidate(
                    base_tensor, pas, pas, pdp, pdp, spec
                )
                values = point_metrics(
                    prediction, truth, truth_pas, truth_pdp
                )
            for key, value in zip(("pas", "pdp", "error", "energy"), values):
                metrics[key][index, start:end] = value
        print(f"V10 full-channel {end}/{len(query_ids)}", flush=True)
    rows = []
    for index, (pas_name, pdp_name, alpha_pas, alpha_pdp) in enumerate(combinations):
        overall = statistics(
            metrics["pas"][index],
            metrics["pdp"][index],
            metrics["error"][index],
            metrics["energy"][index],
        )
        folds = []
        for fold in range(5):
            folds.append(
                statistics(
                    metrics["pas"][index],
                    metrics["pdp"][index],
                    metrics["error"][index],
                    metrics["energy"][index],
                    fold_id == fold,
                )
            )
        rows.append(
            {
                "pas_field": pas_name,
                "pdp_field": pdp_name,
                "alpha_pas": alpha_pas,
                "alpha_pdp": alpha_pdp,
                "overall": overall,
                "folds": folds,
            }
        )
    rows.sort(key=lambda row: float(row["overall"]["score"]), reverse=True)
    serializable_metrics = {
        **metrics,
        "candidate_names": np.asarray(
            [
                f"{pas}_{pdp}_ap{ap:.2f}_ad{ad:.2f}"
                for pas, pdp, ap, ad in combinations
            ]
        ),
        "query_ids": query_ids,
        "fold_id": fold_id,
    }
    return {"selected": rows[0], "top20": rows[:20]}, serializable_metrics


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (OLD_PAS, OLD_PDP, JOINT_PAS, JOINT_PDP, TRUTH_PAS, TRUTH_PDP)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, positions, _, site, _ = load_positions()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    pas_analysis, pas_outputs = prepare_domain(
        "pas", device, args.field_batch_size, query_ids, fold_id, positions, site
    )
    pdp_analysis, pdp_outputs = prepare_domain(
        "pdp", device, args.field_batch_size, query_ids, fold_id, positions, site
    )
    full_analysis, metrics = evaluate_full_channel(
        pas_outputs, pdp_outputs, device, args.batch_size, query_ids, fold_id
    )
    np.savez_compressed(METRICS, **metrics)
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The official-minus-calibrated-Sionna spectral residual is spatially "
            "coherent across independent holes and can correct the V9 endpoint itself, "
            "rather than merely retuning projection strength."
        ),
        "strict_protocol": (
            "Every query uses residual donors from different spatial folds and the same "
            "base-station site; no query truth enters its own residual field."
        ),
        "pas_endpoint": pas_analysis,
        "pdp_endpoint": pdp_analysis,
        "full_channel": full_analysis,
        "artifacts": {"metrics": PATHS.relative(METRICS)},
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
