#!/usr/bin/env python3
"""Evaluate point-joint Sionna endpoints on the Round 2 V7 carrier."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
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
from prepare_sionna_calibration_dataset import (
    ARTIFACT_DIR as E40_DIR,
    TRUTH_PAS,
    TRUTH_PDP,
)
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META, statistics
from run_v8_full_sionna_migration import reconstruct_v7_nested
from train_v9_joint_sionna_endpoints import output_path as joint_output_path


VERSION = "Round2-E58-V9-joint-Sionna-migration"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v9_joint_sionna_migration"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v9_joint_sionna_migration"
COARSE_METRICS = ARTIFACT_DIR / "joint_migration_metrics.npz"
COARSE_REPORT = ANALYSIS_DIR / "joint_migration.json"
REFINE_METRICS = ARTIFACT_DIR / "joint_migration_refine_metrics.npz"
REFINE_REPORT = ANALYSIS_DIR / "joint_migration_refine.json"
OLD_PAS = E40_DIR / "oof_pas_mean_f16.npy"
OLD_PDP = E40_DIR / "oof_pdp_mean_f16.npy"
JOINT_PAS = joint_output_path("pas", "oof")
JOINT_PDP = joint_output_path("pdp", "oof")
TINY = 1e-30


@dataclass(frozen=True)
class Candidate:
    name: str
    pas_beta: float = 0.0
    pdp_beta: float = 0.0
    alpha_pas: float = 0.0
    alpha_pdp: float = 0.0
    order: str = "pas_pdp"
    cycles: int = 4


def coarse_candidates() -> list[Candidate]:
    result = [Candidate("identity")]
    for pas_beta in (0.0, 0.25, 0.50, 0.75, 1.0):
        for pdp_beta in (0.0, 0.50, 1.0):
            for alpha_pas in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
                for alpha_pdp in (0.15, 0.30, 0.45):
                    result.append(
                        Candidate(
                            name=(
                                f"pb{int(round(100 * pas_beta)):03d}_"
                                f"db{int(round(100 * pdp_beta)):03d}_"
                                f"ap{int(round(100 * alpha_pas)):03d}_"
                                f"ad{int(round(100 * alpha_pdp)):03d}"
                            ),
                            pas_beta=pas_beta,
                            pdp_beta=pdp_beta,
                            alpha_pas=alpha_pas,
                            alpha_pdp=alpha_pdp,
                        )
                    )
    return result


def refine_candidates() -> list[Candidate]:
    result = [
        Candidate("identity"),
        Candidate(
            "v8_reference",
            pas_beta=0.0,
            pdp_beta=0.0,
            alpha_pas=0.30,
            alpha_pdp=0.30,
        ),
    ]
    for pas_beta in (0.65, 0.70, 0.75, 0.80, 0.85):
        for pdp_beta in (0.35, 0.50, 0.65):
            for alpha_pas in (0.45, 0.50, 0.55):
                for alpha_pdp in (0.25, 0.30, 0.35):
                    result.append(
                        Candidate(
                            name=(
                                f"pb{int(round(100 * pas_beta)):03d}_"
                                f"db{int(round(100 * pdp_beta)):03d}_"
                                f"ap{int(round(100 * alpha_pas)):03d}_"
                                f"ad{int(round(100 * alpha_pdp)):03d}"
                            ),
                            pas_beta=pas_beta,
                            pdp_beta=pdp_beta,
                            alpha_pas=alpha_pas,
                            alpha_pdp=alpha_pdp,
                        )
                    )
    return result


def candidates(grid: str) -> list[Candidate]:
    return coarse_candidates() if grid == "coarse" else refine_candidates()


def output_files(grid: str) -> tuple[Path, Path]:
    if grid == "coarse":
        return COARSE_METRICS, COARSE_REPORT
    return REFINE_METRICS, REFINE_REPORT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grid", choices=("coarse", "refine"), default="coarse")
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def normalize(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), 0.0)


def endpoint_blend(
    old: torch.Tensor, joint: torch.Tensor, beta: float, dims: tuple[int, ...]
) -> torch.Tensor:
    if beta <= 0.0:
        return old
    if beta >= 1.0:
        return joint
    return normalize((1.0 - beta) * old + beta * joint, dims)


def apply_candidate(
    base: torch.Tensor,
    old_pas: torch.Tensor,
    joint_pas: torch.Tensor,
    old_pdp: torch.Tensor,
    joint_pdp: torch.Tensor,
    spec: Candidate,
) -> torch.Tensor:
    if spec.name == "identity":
        return base
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    pas_target = endpoint_blend(old_pas, joint_pas, spec.pas_beta, (1, 2, 3))
    pdp_target = endpoint_blend(old_pdp, joint_pdp, spec.pdp_beta, (3,))
    desired_pas = fixed_blend(
        pas_distribution(base), pas_target, spec.alpha_pas, (1, 2, 3)
    )
    desired_pdp = fixed_blend(
        pdp_distribution(base), pdp_target, spec.alpha_pdp, (3,)
    )
    candidate = base
    for _ in range(spec.cycles):
        for domain in spec.order.split("_"):
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
    pas = (pas_num / torch.where(pas_den == 0.0, 1e-12, pas_den)).mean(
        dim=(1, 2), dtype=torch.float64
    )
    pdp_num = (prediction_pdp * truth_pdp).sum(dim=3, dtype=torch.float64)
    pdp_den = torch.sqrt(
        prediction_pdp.square().sum(dim=3, dtype=torch.float64)
        * truth_pdp.square().sum(dim=3, dtype=torch.float64)
    )
    pdp = (pdp_num / torch.where(pdp_den == 0.0, 1e-12, pdp_den)).mean(
        dim=(1, 2), dtype=torch.float64
    )
    error = (prediction - truth).abs().square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    energy = truth.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    return tuple(value.cpu().numpy() for value in (pas, pdp, error, energy))


def endpoint_cosine(
    prediction: torch.Tensor, truth: torch.Tensor, domain: str
) -> np.ndarray:
    axes = (1, 2, 3) if domain == "pas" else (3,)
    average = (1, 2)
    numerator = (prediction * truth).sum(dim=axes, dtype=torch.float64)
    denominator = torch.sqrt(
        prediction.square().sum(dim=axes, dtype=torch.float64)
        * truth.square().sum(dim=axes, dtype=torch.float64)
    )
    return (
        numerator / torch.where(denominator == 0.0, 1e-12, denominator)
    ).mean(dim=average, dtype=torch.float64).cpu().numpy()


def generate_metrics(
    device: torch.device, batch_size: int, grid: str, metrics_path: Path
) -> dict[str, np.ndarray]:
    base = reconstruct_v7_nested(device, batch_size, True)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    specs = candidates(grid)
    result = {
        key: np.empty((len(specs), len(query_ids)), dtype=np.float64)
        for key in ("pas", "pdp", "error", "energy")
    }
    direct = {
        key: np.empty((5, len(query_ids)), dtype=np.float64)
        for key in ("pas", "pdp")
    }
    truth_channel = np.load(
        PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r"
    )
    truth_pas_all = np.load(TRUTH_PAS, mmap_mode="r")
    truth_pdp_all = np.load(TRUTH_PDP, mmap_mode="r")
    old_pas_all = np.load(OLD_PAS, mmap_mode="r")
    old_pdp_all = np.load(OLD_PDP, mmap_mode="r")
    joint_pas_all = np.load(JOINT_PAS, mmap_mode="r")
    joint_pdp_all = np.load(JOINT_PDP, mmap_mode="r")
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(
                truth_channel[query_ids[start:end]], dtype=np.complex64, copy=True
            ),
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
        old_pas = torch.as_tensor(
            np.asarray(old_pas_all[start:end], dtype=np.float32), device=device
        )
        old_pdp = torch.as_tensor(
            np.asarray(old_pdp_all[start:end], dtype=np.float32), device=device
        )
        joint_pas = torch.as_tensor(
            np.asarray(joint_pas_all[start:end], dtype=np.float32), device=device
        )
        joint_pdp = torch.as_tensor(
            np.asarray(joint_pdp_all[start:end], dtype=np.float32), device=device
        )
        with torch.inference_mode():
            for beta_index, beta in enumerate((0.0, 0.25, 0.50, 0.75, 1.0)):
                direct["pas"][beta_index, start:end] = endpoint_cosine(
                    endpoint_blend(old_pas, joint_pas, beta, (1, 2, 3)),
                    truth_pas,
                    "pas",
                )
                direct["pdp"][beta_index, start:end] = endpoint_cosine(
                    endpoint_blend(old_pdp, joint_pdp, beta, (3,)),
                    truth_pdp,
                    "pdp",
                )
            truth_pas_view, truth_pdp_view = truth_views(truth)
            for index, spec in enumerate(specs):
                prediction = apply_candidate(
                    base_tensor,
                    old_pas,
                    joint_pas,
                    old_pdp,
                    joint_pdp,
                    spec,
                )
                values = point_metrics(
                    prediction, truth, truth_pas_view, truth_pdp_view
                )
                for key, value in zip(("pas", "pdp", "error", "energy"), values):
                    result[key][index, start:end] = value
        print(f"V9 joint migration grid {end}/{len(query_ids)}", flush=True)
    result.update(
        {
            "candidate_names": np.asarray([spec.name for spec in specs], dtype="U48"),
            "query_ids": query_ids,
            "fold_id": fold_id,
            "direct_pas": direct["pas"],
            "direct_pdp": direct["pdp"],
            "direct_betas": np.asarray((0.0, 0.25, 0.50, 0.75, 1.0)),
        }
    )
    np.savez_compressed(metrics_path, **result)
    return result


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


def fold_deltas(
    metrics: dict[str, np.ndarray], index: int, baseline_index: int
) -> list[dict[str, object]]:
    rows = []
    for fold in np.unique(metrics["fold_id"]):
        mask = metrics["fold_id"] == fold
        current = candidate_result(metrics, index, mask)
        baseline = candidate_result(metrics, baseline_index, mask)
        rows.append(
            {
                "fold": int(fold),
                "result": current,
                "delta_vs_v8": float(current["score"] - baseline["score"]),
            }
        )
    return rows


def analyze(metrics: dict[str, np.ndarray], grid: str) -> dict[str, object]:
    names = metrics["candidate_names"].astype(str)
    baseline_name = (
        "pb000_db000_ap030_ad030" if grid == "coarse" else "v8_reference"
    )
    baseline_index = int(np.flatnonzero(names == baseline_name)[0])
    rows = []
    for index, name in enumerate(names):
        result = candidate_result(metrics, index)
        folds = fold_deltas(metrics, index, baseline_index)
        rows.append(
            {
                "candidate": name,
                **result,
                "delta_vs_v8": float(
                    result["score"]
                    - candidate_result(metrics, baseline_index)["score"]
                ),
                "positive_folds_vs_v8": int(
                    sum(row["delta_vs_v8"] > 0.0 for row in folds)
                ),
                "worst_fold_delta_vs_v8": float(
                    min(row["delta_vs_v8"] for row in folds)
                ),
                "folds": folds,
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    deployable = [
        row
        for row in rows
        if row["candidate"] != "identity"
        and int(row["positive_folds_vs_v8"]) >= 4
        and float(row["worst_fold_delta_vs_v8"]) >= -0.002
    ]
    selected = deployable[0] if deployable else rows[0]
    direct = {}
    for index, beta in enumerate(metrics["direct_betas"]):
        direct[f"beta_{float(beta):.2f}"] = {
            "pas": float(np.mean(metrics["direct_pas"][index])),
            "pdp": float(np.mean(metrics["direct_pdp"][index])),
            "pas_folds": [
                float(
                    np.mean(
                        metrics["direct_pas"][index][metrics["fold_id"] == fold]
                    )
                )
                for fold in range(5)
            ],
            "pdp_folds": [
                float(
                    np.mean(
                        metrics["direct_pdp"][index][metrics["fold_id"] == fold]
                    )
                )
                for fold in range(5)
            ],
        }
    return {
        "v7_identity": candidate_result(metrics, 0),
        "v8_reference": {
            "candidate": baseline_name,
            **candidate_result(metrics, baseline_index),
        },
        "direct_endpoint_grid": direct,
        "selected": selected,
        "top_candidates": rows[:20],
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (OLD_PAS, OLD_PDP, JOINT_PAS, JOINT_PDP)
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing endpoint files: {missing}")
    metrics_path, report_path = output_files(args.grid)
    if args.reuse and metrics_path.is_file():
        metrics = {key: value for key, value in np.load(metrics_path).items()}
    else:
        metrics = generate_metrics(
            torch.device(args.device), args.batch_size, args.grid, metrics_path
        )
    analysis = analyze(metrics, args.grid)
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The point-joint PAS endpoint contains deployable information missing from "
            "E40, allowing stronger scorer-aligned correction on the V7 carrier; the "
            "joint PDP endpoint is retained only where full-channel evidence supports it."
        ),
        "baseline": "V8 old PAS/PDP beta=0, alpha=0.30/0.30",
        "grid": args.grid,
        "candidate_count": int(len(metrics["candidate_names"])),
        "candidates": [asdict(spec) for spec in candidates(args.grid)],
        "analysis": analysis,
        "artifacts": {"metrics": PATHS.relative(metrics_path)},
        "seconds": float(time.time() - started),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
