#!/usr/bin/env python3
"""Measure linkwise donor-selection ceilings hidden by point-global V2/V3 weights."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from project_paths import PATHS


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_linkwise_donor"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_linkwise_donor"
REPORT = ANALYSIS_DIR / "linkwise_donor_oracle.json"
POINT_CACHE = ARTIFACT_DIR / "linkwise_donor_oracle_points.npz"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
TRAIN_PAS = PRE_DIR / "round2_train_pas_link_support_f16.npy"
TRAIN_PDP = PRE_DIR / "round2_train_pdp_link_support_f16.npy"
OOF_WEIGHTS = (
    PATHS.artifacts / "round2" / "rbf_spectral_field" / "round2_rbf_oof_weights.npz"
)

PAS_BLOCKS = (192, 48, 24, 12, 6, 1)
PDP_ANTENNA_BLOCKS = (256, 128, 64, 32, 16, 8, 1)


def cosine_axis(left: torch.Tensor, right: torch.Tensor, dim: int) -> torch.Tensor:
    numerator = torch.sum(left * right, dim=dim)
    denominator = torch.sqrt(
        torch.sum(left.square(), dim=dim) * torch.sum(right.square(), dim=dim)
    )
    return torch.where(
        denominator > 0.0,
        numerator / denominator.clamp_min(1e-20),
        torch.zeros_like(numerator),
    )


def block_oracle(similarity: torch.Tensor, axis: int, block: int) -> torch.Tensor:
    # Similarity starts with donor dimension and contains one official score per link.
    moved = similarity.movedim(axis, -1)
    length = moved.shape[-1]
    if length % block != 0:
        raise ValueError((length, block))
    shaped = moved.reshape(*moved.shape[:-1], length // block, block)
    block_score = shaped.mean(dim=-1)
    selected = torch.argmax(block_score, dim=0)
    selected_score = torch.gather(
        block_score, 0, selected.unsqueeze(0)
    ).squeeze(0)
    return selected_score.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (TRAIN_PAS, TRAIN_PDP, OOF_WEIGHTS):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    pas_cache = np.load(TRAIN_PAS, mmap_mode="r")
    pdp_cache = np.load(TRAIN_PDP, mmap_mode="r")
    weights = np.load(OOF_WEIGHTS)
    query_ids = weights["query_ids"].astype(np.int64)
    fold_id = weights["fold_id"].astype(np.int8)
    donor_ids = weights["neighbor_ids"].astype(np.int64)
    base_weights = weights["weights"].astype(np.float32)

    names = (
        ["pas_base", "pas_point_oracle", "pas_link_oracle"]
        + [f"pas_frequency_block_{block}" for block in PAS_BLOCKS]
        + ["pdp_base", "pdp_point_oracle", "pdp_link_oracle"]
        + [f"pdp_antenna_block_{block}" for block in PDP_ANTENNA_BLOCKS]
    )
    point = {name: np.empty(len(query_ids), dtype=np.float32) for name in names}

    for row, (query_id, donors, row_weights) in enumerate(
        zip(query_ids, donor_ids, base_weights)
    ):
        donor_pas = torch.as_tensor(
            np.asarray(pas_cache[donors], dtype=np.float32), device=device
        )
        truth_pas = torch.as_tensor(
            np.asarray(pas_cache[query_id], dtype=np.float32), device=device
        )
        amplitude_pas = torch.sqrt(donor_pas.clamp_min(0.0))
        predicted_pas = torch.relu(
            torch.sum(
                torch.as_tensor(row_weights, device=device)[:, None, None, None]
                * amplitude_pas,
                dim=0,
            )
        ).square()
        predicted_pas = predicted_pas / predicted_pas.sum(dim=0, keepdim=True).clamp_min(1e-20)
        point["pas_base"][row] = float(
            cosine_axis(predicted_pas, truth_pas, 0).mean().cpu()
        )
        pas_similarity = cosine_axis(donor_pas, truth_pas[None], 1)
        point["pas_point_oracle"][row] = float(
            torch.max(pas_similarity.mean(dim=(1, 2))).cpu()
        )
        point["pas_link_oracle"][row] = float(
            pas_similarity.max(dim=0).values.mean().cpu()
        )
        for block in PAS_BLOCKS:
            point[f"pas_frequency_block_{block}"][row] = float(
                block_oracle(pas_similarity, axis=2, block=block).cpu()
            )

        donor_pdp = torch.as_tensor(
            np.asarray(pdp_cache[donors], dtype=np.float32), device=device
        )
        truth_pdp = torch.as_tensor(
            np.asarray(pdp_cache[query_id], dtype=np.float32), device=device
        )
        amplitude_pdp = torch.sqrt(donor_pdp.clamp_min(0.0))
        predicted_pdp = torch.relu(
            torch.sum(
                torch.as_tensor(row_weights, device=device)[:, None, None, None]
                * amplitude_pdp,
                dim=0,
            )
        ).square()
        predicted_pdp = predicted_pdp / predicted_pdp.sum(dim=2, keepdim=True).clamp_min(1e-20)
        point["pdp_base"][row] = float(
            cosine_axis(predicted_pdp, truth_pdp, 2).mean().cpu()
        )
        pdp_similarity = cosine_axis(donor_pdp, truth_pdp[None], 3)
        point["pdp_point_oracle"][row] = float(
            torch.max(pdp_similarity.mean(dim=(1, 2))).cpu()
        )
        point["pdp_link_oracle"][row] = float(
            pdp_similarity.max(dim=0).values.mean().cpu()
        )
        for block in PDP_ANTENNA_BLOCKS:
            point[f"pdp_antenna_block_{block}"][row] = float(
                block_oracle(pdp_similarity, axis=1, block=block).cpu()
            )
        if (row + 1) % 25 == 0 or row + 1 == len(query_ids):
            print(f"linkwise oracle {row + 1}/{len(query_ids)}", flush=True)

    np.savez_compressed(
        POINT_CACHE,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        **point,
    )
    means = {name: float(np.mean(values)) for name, values in point.items()}
    fold_means = {
        name: [
            float(np.mean(values[fold_id == fold])) for fold in np.unique(fold_id)
        ]
        for name, values in point.items()
    }
    report = {
        "experiment": "Round2-E12-linkwise-donor-oracle",
        "status": "analysis_only",
        "primary_hypothesis": (
            "A single donor mixture for every link suppresses official linkwise PAS/PDP "
            "accuracy; low-cardinality group-conditioned donor selection has a much higher ceiling."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "donors": int(donor_ids.shape[1]),
        "means": means,
        "fold_means": fold_means,
        "deltas": {
            "pas_point_oracle_vs_base": means["pas_point_oracle"] - means["pas_base"],
            "pas_link_oracle_vs_base": means["pas_link_oracle"] - means["pas_base"],
            "pdp_point_oracle_vs_base": means["pdp_point_oracle"] - means["pdp_base"],
            "pdp_link_oracle_vs_base": means["pdp_link_oracle"] - means["pdp_base"],
        },
        "outputs": {
            "point_cache": PATHS.relative(POINT_CACHE),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
