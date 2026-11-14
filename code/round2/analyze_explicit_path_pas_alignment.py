#!/usr/bin/env python3
"""Recover fixed array-domain alignment for the explicit-path PAS source."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from analyze_joint_path_token_oracle import OOF_META
from analyze_v11_explicit_path_pas_upper_bound import RAW_PAS, normalize_pas
from analyze_v4_exact_joint_projection import truth_views
from project_paths import PATHS
from round2.data import load_positions


VERSION = "Round2-E72-explicit-path-PAS-array-alignment"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "explicit_path_pas_alignment"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_explicit_path_pas_alignment"
ALIGNED_OOF = ARTIFACT_DIR / "oof_aligned_raw_path_pas_f16.npy"
REPORT = ANALYSIS_DIR / "explicit_path_pas_alignment.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def cosine_slices(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=(1, 2, 3), dtype=torch.float64)
    denominator = torch.sqrt(
        prediction.square().sum(dim=(1, 2, 3), dtype=torch.float64)
        * truth.square().sum(dim=(1, 2, 3), dtype=torch.float64)
    )
    return numerator / torch.where(denominator == 0.0, 1e-12, denominator)


def transform(
    value: torch.Tensor,
    h_shift: int,
    v_shift: int,
    h_flip: bool,
    v_flip: bool,
    pol_swap: bool,
) -> torch.Tensor:
    output = value
    if h_flip:
        output = torch.flip(output, dims=(2,))
    if v_flip:
        output = torch.flip(output, dims=(3,))
    if pol_swap:
        output = torch.flip(output, dims=(1,))
    return torch.roll(output, shifts=(h_shift, v_shift), dims=(2, 3))


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_PAS.is_file():
        raise FileNotFoundError(PATHS.relative(RAW_PAS))
    device = torch.device(args.device)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_all = metadata["fold_id"].astype(np.int8)
    path_cache = (
        PATHS.artifacts
        / "round2"
        / "joint_path_token_oracle"
        / "n469_s24_k128_b096_d064_scatter1_sel106"
    )
    rows = np.load(path_cache / "rows.npy").astype(np.int64)
    global_ids = query_ids[rows]
    fold_id = fold_all[rows]
    _, _, _, train_site, _ = load_positions()
    site = train_site[global_ids]
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth = torch.as_tensor(
        np.asarray(truth_all[global_ids], dtype=np.complex64), device=device
    )
    truth_pas, _ = truth_views(truth)
    truth_pas = normalize_pas(truth_pas)
    raw = normalize_pas(
        torch.as_tensor(np.asarray(np.load(RAW_PAS, mmap_mode="r"), dtype=np.float32), device=device)
    )
    baseline = float(cosine_slices(raw, truth_pas).mean().item())

    winners = {}
    aligned = raw.clone()
    for site_value in (0, 1):
        mask = torch.as_tensor(site == site_value, device=device)
        local_raw = raw[mask]
        local_truth = truth_pas[mask]
        rows_out = []
        for h_flip in (False, True):
            for v_flip in (False, True):
                for pol_swap in (False, True):
                    oriented = transform(
                        local_raw, 0, 0, h_flip, v_flip, pol_swap
                    )
                    for h_shift in range(-4, 5):
                        for v_shift in range(-4, 5):
                            candidate = torch.roll(
                                oriented,
                                shifts=(h_shift, v_shift),
                                dims=(2, 3),
                            )
                            score = float(
                                cosine_slices(candidate, local_truth).mean().item()
                            )
                            rows_out.append(
                                {
                                    "h_shift": h_shift,
                                    "v_shift": v_shift,
                                    "h_flip": h_flip,
                                    "v_flip": v_flip,
                                    "pol_swap": pol_swap,
                                    "pas": score,
                                }
                            )
        rows_out.sort(key=lambda row: float(row["pas"]), reverse=True)
        winner = rows_out[0]
        winners[str(site_value)] = {
            "count": int(np.sum(site == site_value)),
            "baseline": float(cosine_slices(local_raw, local_truth).mean().item()),
            "selected": winner,
            "top20": rows_out[:20],
        }
        aligned[mask] = transform(
            local_raw,
            int(winner["h_shift"]),
            int(winner["v_shift"]),
            bool(winner["h_flip"]),
            bool(winner["v_flip"]),
            bool(winner["pol_swap"]),
        )

    aligned_scores = cosine_slices(aligned, truth_pas)
    folds = []
    for fold in range(5):
        mask = torch.as_tensor(fold_id == fold, device=device)
        folds.append(
            {
                "fold": fold,
                "baseline": float(cosine_slices(raw[mask], truth_pas[mask]).mean().item()),
                "selected": float(aligned_scores[mask].mean().item()),
            }
        )
        folds[-1]["delta"] = folds[-1]["selected"] - folds[-1]["baseline"]
    output = np.lib.format.open_memmap(
        ALIGNED_OOF, mode="w+", dtype=np.float16, shape=tuple(aligned.shape)
    )
    output[...] = aligned.cpu().numpy().astype(np.float16)
    output.flush()
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "point_count": int(len(raw)),
        "baseline": baseline,
        "selected": float(aligned_scores.mean().item()),
        "delta": float(aligned_scores.mean().item() - baseline),
        "site_winners": winners,
        "folds": folds,
        "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
        "output": PATHS.relative(ALIGNED_OOF),
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
