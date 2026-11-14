#!/usr/bin/env python3
"""Measure the usable PAS headroom in the explicit paths behind Round 2 V10."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from analyze_joint_path_token_oracle import FREQUENCIES, OOF_META
from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    pas_distribution,
    preserve_point_power,
    truth_views,
)
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics
from train_path_pdp_link_gate import project


VERSION = "Round2-E68-explicit-path-PAS-upper-bound"
PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n469_s24_k128_b096_d064_scatter1_sel106"
)
PDP_DIR = PATHS.artifacts / "round2" / "path_pdp_link_gate"
V10_DIR = PATHS.artifacts / "round2" / "full_path_pdp_link_gate"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "explicit_path_pas_joint_gate"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_explicit_path_pas_joint_gate"
RAW_PAS = ARTIFACT_DIR / "oof_raw_path_pas_f16.npy"
REPORT = ANALYSIS_DIR / "explicit_path_pas_upper_bound.json"
ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def cosine_slices(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=(1, 2, 3), dtype=torch.float64)
    denominator = torch.sqrt(
        prediction.square().sum(dim=(1, 2, 3), dtype=torch.float64)
        * truth.square().sum(dim=(1, 2, 3), dtype=torch.float64)
    )
    return numerator / torch.where(denominator == 0.0, 1e-12, denominator)


def normalize_pas(value: torch.Tensor) -> torch.Tensor:
    total = value.sum(dim=(1, 2, 3), keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), 0.0)


def build_raw_pas(args: argparse.Namespace, device: torch.device) -> np.ndarray:
    if RAW_PAS.is_file() and not args.force:
        return np.load(RAW_PAS, mmap_mode="r")
    coefficient = np.load(PATH_CACHE / "coefficient.npy", mmap_mode="r")
    tau = np.load(PATH_CACHE / "tau.npy", mmap_mode="r")
    mask = np.load(PATH_CACHE / "mask.npy", mmap_mode="r")
    output = np.lib.format.open_memmap(
        RAW_PAS,
        mode="w+",
        dtype=np.float16,
        shape=(len(coefficient), 2, 16, 8, 4, 192),
    )
    frequency = torch.as_tensor(FREQUENCIES, device=device, dtype=torch.float32)
    for start in range(0, len(coefficient), args.batch_size):
        end = min(start + args.batch_size, len(coefficient))
        coefficient_tensor = torch.as_tensor(
            np.array(coefficient[start:end], copy=True), device=device
        )
        tau_tensor = torch.as_tensor(
            np.array(tau[start:end], dtype=np.float32, copy=True), device=device
        )
        mask_tensor = torch.as_tensor(
            np.asarray(mask[start:end]), device=device
        )
        phase = torch.exp(
            -2j * torch.pi * tau_tensor[:, :, None] * frequency[None, None]
        )
        channel = torch.einsum(
            "bk,bkmu,bks->bmus",
            mask_tensor.to(coefficient_tensor.dtype),
            coefficient_tensor,
            phase,
        )
        output[start:end] = pas_distribution(channel).cpu().numpy().astype(np.float16)
        if end % 100 == 0 or end == len(coefficient):
            output.flush()
            print(f"raw path PAS {end}/{len(coefficient)}", flush=True)
    output.flush()
    return np.load(RAW_PAS, mmap_mode="r")


def v10_oof_channel(device: torch.device) -> np.ndarray:
    base = np.load(PDP_DIR / "v9_base_channel.npy", mmap_mode="r")
    base_pdp = np.load(PDP_DIR / "v9_base_pdp_f16.npy", mmap_mode="r")
    raw_pdp = np.load(PDP_DIR / "raw_path_pdp_f16.npy", mmap_mode="r")
    alpha = np.asarray(
        np.load(V10_DIR / "v9_oof_pretrained_alpha_f16.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    desired = (
        (1.0 - np.clip(0.5 * alpha, 0.0, 1.0)[..., None])
        * np.asarray(base_pdp, dtype=np.float32).reshape(len(base), 1024, 192)
        + np.clip(0.5 * alpha, 0.0, 1.0)[..., None]
        * np.asarray(raw_pdp, dtype=np.float32).reshape(len(base), 1024, 192)
    )
    desired /= np.maximum(desired.sum(axis=2, keepdims=True), TINY)
    with torch.inference_mode():
        result = project(
            torch.as_tensor(np.asarray(base), device=device),
            torch.as_tensor(desired.reshape(len(base), 256, 4, 192), device=device),
            2,
            "pas_pdp",
        )
    return result.cpu().numpy().astype(np.complex64)


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (
        PATH_CACHE,
        PDP_DIR / "v9_base_channel.npy",
        PDP_DIR / "v9_base_pdp_f16.npy",
        PDP_DIR / "raw_path_pdp_f16.npy",
        V10_DIR / "v9_oof_pretrained_alpha_f16.npy",
        OOF_META,
    )
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    rows = np.load(PATH_CACHE / "rows.npy").astype(np.int64)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)[rows]
    global_ids = query_ids[rows]
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth = np.asarray(truth_all[global_ids], dtype=np.complex64)
    base = v10_oof_channel(device)
    raw_pas_array = build_raw_pas(args, device)

    base_tensor = torch.as_tensor(base, device=device)
    truth_tensor = torch.as_tensor(truth, device=device)
    truth_pas, truth_pdp = truth_views(truth_tensor)
    base_pas = pas_distribution(base_tensor)
    raw_pas = torch.as_tensor(
        np.asarray(raw_pas_array, dtype=np.float32), device=device
    )
    raw_pas = normalize_pas(raw_pas)
    truth_pas_normalized = normalize_pas(truth_pas)

    endpoint_rows = []
    slice_scores = []
    point_scores = []
    desired_bank = []
    for alpha in ALPHAS:
        desired = normalize_pas((1.0 - alpha) * base_pas + alpha * raw_pas)
        scores = cosine_slices(desired, truth_pas_normalized)
        slice_scores.append(scores)
        point_scores.append(scores.mean(dim=(1, 2), dtype=torch.float64))
        desired_bank.append(desired)
        endpoint_rows.append(
            {
                "alpha": alpha,
                "pas": float(scores.mean(dtype=torch.float64).item()),
            }
        )

    slice_stack = torch.stack(slice_scores, dim=0)
    point_stack = torch.stack(point_scores, dim=0)
    point_choice = torch.argmax(point_stack, dim=0)
    slice_choice = torch.argmax(slice_stack, dim=0)
    desired_stack = torch.stack(desired_bank, dim=0)
    point_index = point_choice[None, :, None, None, None, None, None].expand(
        1, len(base), 2, 16, 8, 4, 192
    )
    point_desired = torch.gather(desired_stack, 0, point_index)[0]
    slice_index = slice_choice[None, :, None, None, None, :, :].expand(
        1, len(base), 2, 16, 8, 4, 192
    )
    slice_desired = torch.gather(desired_stack, 0, slice_index)[0]

    baseline_values = point_metrics(base_tensor, truth_tensor, truth_pas, truth_pdp)
    projected = {}
    for name, desired in (
        ("point_oracle", point_desired),
        ("slice_oracle", slice_desired),
    ):
        with torch.inference_mode():
            candidate = exact_pas_projection(base_tensor, desired)
            reference_power = base_tensor.abs().square().sum(dim=(1, 2, 3))
            candidate = preserve_point_power(candidate, reference_power)
            values = point_metrics(candidate, truth_tensor, truth_pas, truth_pdp)
        projected[name] = {
            "overall": statistics(*values),
            "folds": [
                {
                    "fold": fold,
                    "baseline": statistics(*baseline_values, fold_id == fold),
                    "result": statistics(*values, fold_id == fold),
                }
                for fold in range(5)
            ],
        }
        for row in projected[name]["folds"]:
            row["delta"] = float(
                row["result"]["score"] - row["baseline"]["score"]
            )

    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "point_count": int(len(base)),
        "baseline": statistics(*baseline_values),
        "raw_path_pas": float(
            cosine_slices(raw_pas, truth_pas_normalized).mean(dtype=torch.float64).item()
        ),
        "global_alpha_grid": endpoint_rows,
        "point_oracle_pas": float(
            torch.max(point_stack, dim=0).values.mean(dtype=torch.float64).item()
        ),
        "slice_oracle_pas": float(
            torch.max(slice_stack, dim=0).values.mean(dtype=torch.float64).item()
        ),
        "point_choice_histogram": {
            str(alpha): int(torch.sum(point_choice == index).item())
            for index, alpha in enumerate(ALPHAS)
        },
        "slice_choice_histogram": {
            str(alpha): int(torch.sum(slice_choice == index).item())
            for index, alpha in enumerate(ALPHAS)
        },
        "projected": projected,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
