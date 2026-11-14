#!/usr/bin/env python3
"""Generate Round 2 V8 by applying calibrated Sionna PAS/PDP to the V7 carrier."""

from __future__ import annotations

import argparse
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
)
from prepare_sionna_calibration_dataset import ARTIFACT_DIR as CALIBRATION_DIR
from project_paths import PATHS


VERSION = "V8"
METHOD = "full_sionna_migration"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "full_sionna_migration"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_full_sionna_migration"
REPORT = ANALYSIS_DIR / "v8_generate.json"
V7_TEST = (
    PATHS.artifacts
    / "round2"
    / "complex_carrier_lifecycle_overlay"
    / "Round2_V7_Stacked_Lifecycle_Test_Channel.npy"
)
TEST_PAS_TARGET = CALIBRATION_DIR / "test_pas_mean_ensemble_f16.npy"
TEST_PDP_TARGET = CALIBRATION_DIR / "test_pdp_mean_ensemble_f16.npy"
OUTPUT = ARTIFACT_DIR / "Round2_V8_Full_Sionna_Migration_Test_Channel.npy"

ALPHA_PAS = 0.30
ALPHA_PDP = 0.30
CYCLES = 4
TINY = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def project(device: torch.device, batch_size: int) -> dict[str, object]:
    base = np.load(V7_TEST, mmap_mode="r")
    pas_target = np.load(TEST_PAS_TARGET, mmap_mode="r")
    pdp_target = np.load(TEST_PDP_TARGET, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    relative_change = np.empty(len(base), dtype=np.float64)
    power_ratio = np.empty(len(base), dtype=np.float64)
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        reference = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        target_pas = torch.as_tensor(
            np.asarray(pas_target[start:end], dtype=np.float32), device=device
        )
        target_pdp = torch.as_tensor(
            np.asarray(pdp_target[start:end], dtype=np.float32), device=device
        )
        with torch.inference_mode():
            reference_power = reference.abs().square().sum(dim=(1, 2, 3))
            desired_pas = fixed_blend(
                pas_distribution(reference), target_pas, ALPHA_PAS, (1, 2, 3)
            )
            desired_pdp = fixed_blend(
                pdp_distribution(reference), target_pdp, ALPHA_PDP, (3,)
            )
            candidate = reference
            for _ in range(CYCLES):
                candidate = exact_pas_projection(candidate, desired_pas)
                candidate = exact_pdp_projection(candidate, desired_pdp)
                candidate = preserve_point_power(candidate, reference_power)
            candidate_power = candidate.abs().square().sum(dim=(1, 2, 3))
            delta = (candidate - reference).abs().square().sum(dim=(1, 2, 3))
            valid = reference_power > TINY
            relative_change[start:end] = torch.where(
                valid,
                delta / reference_power.clamp_min(TINY),
                torch.zeros_like(reference_power),
            ).cpu().numpy()
            power_ratio[start:end] = torch.where(
                valid,
                candidate_power / reference_power.clamp_min(TINY),
                torch.ones_like(reference_power),
            ).cpu().numpy()
        output[start:end] = candidate.cpu().numpy().astype(np.complex64)
        print(f"V8 projection {end}/{len(base)}", flush=True)
    output.flush()
    return {
        "path": PATHS.relative(OUTPUT),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(np.isfinite(output.real).all() and np.isfinite(output.imag).all()),
        "bytes": int(OUTPUT.stat().st_size),
        "relative_change_mean": float(np.mean(relative_change)),
        "relative_change_median": float(np.median(relative_change)),
        "relative_change_p90": float(np.quantile(relative_change, 0.90)),
        "power_ratio_mean": float(np.mean(power_ratio)),
        "power_ratio_max_abs_error": float(np.max(np.abs(power_ratio - 1.0))),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    for path in (V7_TEST, TEST_PAS_TARGET, TEST_PDP_TARGET):
        if not path.is_file():
            raise FileNotFoundError(path)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    output = project(device, args.batch_size)
    payload = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_generated_not_packaged",
        "base": PATHS.relative(V7_TEST),
        "policy": {
            "alpha_pas": ALPHA_PAS,
            "alpha_pdp": ALPHA_PDP,
            "order": "pas_pdp",
            "cycles": CYCLES,
            "absolute_sionna_phase_used": False,
            "point_power_preserved": True,
        },
        "targets": {
            "pas": PATHS.relative(TEST_PAS_TARGET),
            "pdp": PATHS.relative(TEST_PDP_TARGET),
        },
        "output": output,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
