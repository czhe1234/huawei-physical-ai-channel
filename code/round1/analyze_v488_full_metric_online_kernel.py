#!/usr/bin/env python3
"""Build a nested-validated online kernel from the complete official metric."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

import analyze_v479_online_spectral_target_inversion as v479
import analyze_v487_augmented_online_kernel as v487
import train_v483_fullbatch_online_kernel_mixer as v483
import train_v485_direct_online_kernel_channel as v485
import train_v486_direct_online_kernel_continuation as v486
from project_paths import PATHS


VERSION_NUMBER = 488
VERSION = "v488_full_metric_online_kernel"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
RIDGES = np.logspace(-8, -1, 15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--point-batch", type=int, default=10)
    return parser.parse_args()


@torch.no_grad()
def pairwise_nmse_score(
    sources: torch.Tensor,
    point_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    count, points = sources.shape[:2]
    error = np.zeros((count, count), dtype=np.float64)
    energy = np.zeros(count, dtype=np.float64)
    for start in range(0, points, point_batch):
        end = min(start + point_batch, points)
        flat = sources[:, start:end].reshape(count, -1)
        block_energy = torch.sum(torch.abs(flat) ** 2, dim=1)
        gram = flat.conj() @ flat.transpose(0, 1)
        block_error = (
            block_energy[:, None]
            + block_energy[None, :]
            - 2.0 * torch.real(gram)
        ).clamp_min(0.0)
        error += block_error.cpu().numpy().astype(np.float64)
        energy += block_energy.cpu().numpy().astype(np.float64)
        if end % 100 == 0:
            print(f"v488 pairwise NMSE {end}/{points}", flush=True)
    nmse = error / np.maximum(energy[None, :], 1e-30)
    return 1.0 / (1.0 + nmse), energy


@torch.no_grad()
def candidate_nmse_score(
    candidate: np.ndarray,
    sources: torch.Tensor,
    source_energy: np.ndarray,
    device: torch.device,
    point_batch: int,
) -> np.ndarray:
    count, points = sources.shape[:2]
    error = np.zeros(count, dtype=np.float64)
    for start in range(0, points, point_batch):
        end = min(start + point_batch, points)
        value = torch.from_numpy(
            np.asarray(candidate[start:end], dtype=np.complex64)
        ).to(device)
        candidate_flat = value.reshape(-1)
        source_flat = sources[:, start:end].reshape(count, -1)
        candidate_energy = torch.sum(torch.abs(candidate_flat) ** 2)
        block_source_energy = torch.sum(torch.abs(source_flat) ** 2, dim=1)
        cross = source_flat @ torch.conj(candidate_flat)
        block_error = (
            candidate_energy + block_source_energy - 2.0 * torch.real(cross)
        ).clamp_min(0.0)
        error += block_error.cpu().numpy().astype(np.float64)
    nmse = error / np.maximum(source_energy, 1e-30)
    return 1.0 / (1.0 + nmse)


def nested_fixed_kernel(
    kernel: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    predictions = np.empty(len(scores), dtype=np.float64)
    rows: list[dict[str, object]] = []
    for outer in range(len(scores)):
        train_ids = np.flatnonzero(np.arange(len(scores)) != outer)
        train_kernel = kernel[np.ix_(train_ids, train_ids)]
        train_scores = scores[train_ids]
        best = None
        for ridge in RIDGES:
            inner_prediction = v479.loo_predictions(
                train_kernel, train_scores, float(ridge)
            )
            rmse = float(np.sqrt(np.mean((inner_prediction - train_scores) ** 2)))
            correlation = float(pearsonr(inner_prediction, train_scores).statistic)
            key = (rmse, -correlation)
            if best is None or key < best[0]:
                best = (key, float(ridge))
        assert best is not None
        ridge = best[1]
        alpha, intercept = v479.kernel_fit(train_kernel, train_scores, ridge)
        prediction = float(kernel[outer, train_ids] @ alpha + intercept)
        predictions[outer] = prediction
        rows.append(
            {
                "held_out_version": int(v487.VERSIONS[outer][0]),
                "truth": float(scores[outer]),
                "prediction": prediction,
                "selected_ridge": ridge,
                "inner_rmse": float(best[0][0]),
            }
        )
    return predictions, rows


def metrics(
    predictions: np.ndarray,
    scores: np.ndarray,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "rmse": float(np.sqrt(np.mean((predictions - scores) ** 2))),
        "pearson": float(pearsonr(predictions, scores).statistic),
        "spearman": float(spearmanr(predictions, scores).statistic),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    augmented_report = json.loads(v487.REPORT.read_text(encoding="utf-8"))
    old_report = json.loads(v479.REPORT.read_text(encoding="utf-8"))
    old_pas = np.asarray(old_report["pas_gram"], dtype=np.float64)
    old_pdp = np.asarray(old_report["pdp_gram"], dtype=np.float64)
    v364_pas = np.asarray(
        augmented_report["v364_similarity_row"]["pas"], dtype=np.float64
    )
    v364_pdp = np.asarray(
        augmented_report["v364_similarity_row"]["pdp"], dtype=np.float64
    )
    count = len(v487.VERSIONS)
    pas_gram = np.empty((count, count), dtype=np.float64)
    pdp_gram = np.empty((count, count), dtype=np.float64)
    pas_gram[:-1, :-1] = old_pas
    pdp_gram[:-1, :-1] = old_pdp
    pas_gram[-1] = v364_pas
    pas_gram[:, -1] = v364_pas
    pdp_gram[-1] = v364_pdp
    pdp_gram[:, -1] = v364_pdp

    arrays = [np.load(path, mmap_mode="r") for _, _, path in v487.VERSIONS]
    device = torch.device(args.device)
    sources = v483.load_sources(arrays, device)
    nmse_kernel, source_energy = pairwise_nmse_score(sources, args.point_batch)
    full_kernel = 0.4 * pas_gram + 0.4 * pdp_gram + 0.2 * nmse_kernel
    scores = np.asarray([score for _, score, _ in v487.VERSIONS], dtype=np.float64)
    nested_prediction, nested_rows = nested_fixed_kernel(full_kernel, scores)
    nested_metrics = metrics(nested_prediction, scores, nested_rows)

    best = None
    for ridge in RIDGES:
        prediction = v479.loo_predictions(full_kernel, scores, float(ridge))
        row = metrics(prediction, scores, [])
        key = (float(row["rmse"]), -float(row["pearson"]), -float(row["spearman"]))
        if best is None or key < best[0]:
            best = (key, float(ridge), prediction, row)
    assert best is not None
    _, ridge, loo_prediction, loo_metrics = best
    alpha, intercept = v479.kernel_fit(full_kernel, scores, ridge)

    candidate_rows = {}
    for name, path in (("v485", v485.OUTPUT), ("v486", v486.OUTPUT)):
        spectral = augmented_report["candidate_predictions"][name]
        pas_similarity = np.asarray(spectral["pas_similarity"], dtype=np.float64)
        pdp_similarity = np.asarray(spectral["pdp_similarity"], dtype=np.float64)
        nmse_similarity = candidate_nmse_score(
            np.load(path, mmap_mode="r"),
            sources,
            source_energy,
            device,
            args.point_batch,
        )
        similarity = (
            0.4 * pas_similarity + 0.4 * pdp_similarity + 0.2 * nmse_similarity
        )
        candidate_rows[name] = {
            "predicted_online_score": float(similarity @ alpha + intercept),
            "pas_similarity": pas_similarity.tolist(),
            "pdp_similarity": pdp_similarity.tolist(),
            "nmse_score_similarity": nmse_similarity.tolist(),
        }

    passed = (
        nested_metrics["rmse"] <= 0.025
        and nested_metrics["pearson"] >= 0.90
        and nested_metrics["spearman"] >= 0.85
    )
    report = {
        "method": VERSION,
        "status": "signal_passed" if passed else "analysis_only_rejected",
        "primary_hypothesis": (
            "The official 0.2 NMSE-score term explains the v364 spectral-kernel outlier "
            "and yields a transferable complete-metric online kernel."
        ),
        "parameters": vars(args),
        "versions": [
            {"version": int(version), "score": float(score), "path": PATHS.relative(path)}
            for version, score, path in v487.VERSIONS
        ],
        "nested_loo": nested_metrics,
        "ordinary_loo": {
            **loo_metrics,
            "ridge": ridge,
            "predictions": loo_prediction.tolist(),
        },
        "full_fit": {"alpha": alpha.tolist(), "intercept": intercept},
        "kernels": {
            "pas": pas_gram.tolist(),
            "pdp": pdp_gram.tolist(),
            "nmse_score": nmse_kernel.tolist(),
            "full_metric": full_kernel.tolist(),
        },
        "candidate_predictions": candidate_rows,
        "decision": passed,
        "report": PATHS.relative(REPORT),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "nested_loo": nested_metrics,
                "ordinary_loo": report["ordinary_loo"],
                "candidate_predictions": {
                    key: value["predicted_online_score"]
                    for key, value in candidate_rows.items()
                },
                "passed": passed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
