#!/usr/bin/env python3
"""Generate Round 2 V9 from the frozen joint-Sionna endpoint policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time

import numpy as np
import torch

from prepare_sionna_calibration_dataset import ARTIFACT_DIR as E40_DIR
from project_paths import PATHS
from run_v9_joint_sionna_migration import Candidate, apply_candidate
from train_v9_joint_sionna_endpoints import output_path as joint_output_path


VERSION = "V9"
METHOD = "joint_sionna_migration"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v9_joint_sionna_migration"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v9_joint_sionna_migration"
REPORT = ANALYSIS_DIR / "v9_generate.json"
V7_TEST = (
    PATHS.artifacts
    / "round2"
    / "complex_carrier_lifecycle_overlay"
    / "Round2_V7_Stacked_Lifecycle_Test_Channel.npy"
)
V8_TEST = (
    PATHS.artifacts
    / "round2"
    / "full_sionna_migration"
    / "Round2_V8_Full_Sionna_Migration_Test_Channel.npy"
)
OLD_PAS = E40_DIR / "test_pas_mean_ensemble_f16.npy"
OLD_PDP = E40_DIR / "test_pdp_mean_ensemble_f16.npy"
JOINT_PAS = joint_output_path("pas", "test")
JOINT_PDP = joint_output_path("pdp", "test")
OUTPUT = ARTIFACT_DIR / "Round2_V9_Joint_Sionna_Migration_Test_Channel.npy"
POLICY = Candidate(
    name="pb075_db050_ap045_ad035",
    pas_beta=0.75,
    pdp_beta=0.50,
    alpha_pas=0.45,
    alpha_pdp=0.35,
    order="pas_pdp",
    cycles=4,
)
TINY = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def relative_nmse(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    energy = reference.abs().square().sum(dim=(1, 2, 3))
    error = (candidate - reference).abs().square().sum(dim=(1, 2, 3))
    return torch.where(
        energy > TINY,
        error / energy.clamp_min(TINY),
        torch.zeros_like(energy),
    )


def generate(device: torch.device, batch_size: int) -> dict[str, object]:
    base = np.load(V7_TEST, mmap_mode="r")
    v8 = np.load(V8_TEST, mmap_mode="r")
    old_pas = np.load(OLD_PAS, mmap_mode="r")
    old_pdp = np.load(OLD_PDP, mmap_mode="r")
    joint_pas = np.load(JOINT_PAS, mmap_mode="r")
    joint_pdp = np.load(JOINT_PDP, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    change_v7 = np.empty(len(base), dtype=np.float64)
    change_v8 = np.empty(len(base), dtype=np.float64)
    power_ratio = np.empty(len(base), dtype=np.float64)

    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        reference = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        v8_reference = torch.as_tensor(
            np.array(v8[start:end], dtype=np.complex64, copy=True), device=device
        )
        endpoint_old_pas = torch.as_tensor(
            np.asarray(old_pas[start:end], dtype=np.float32), device=device
        )
        endpoint_old_pdp = torch.as_tensor(
            np.asarray(old_pdp[start:end], dtype=np.float32), device=device
        )
        endpoint_joint_pas = torch.as_tensor(
            np.asarray(joint_pas[start:end], dtype=np.float32), device=device
        )
        endpoint_joint_pdp = torch.as_tensor(
            np.asarray(joint_pdp[start:end], dtype=np.float32), device=device
        )
        with torch.inference_mode():
            candidate = apply_candidate(
                reference,
                endpoint_old_pas,
                endpoint_joint_pas,
                endpoint_old_pdp,
                endpoint_joint_pdp,
                POLICY,
            )
            reference_power = reference.abs().square().sum(dim=(1, 2, 3))
            candidate_power = candidate.abs().square().sum(dim=(1, 2, 3))
            change_v7[start:end] = relative_nmse(candidate, reference).cpu().numpy()
            change_v8[start:end] = relative_nmse(candidate, v8_reference).cpu().numpy()
            power_ratio[start:end] = torch.where(
                reference_power > TINY,
                candidate_power / reference_power.clamp_min(TINY),
                torch.ones_like(reference_power),
            ).cpu().numpy()
        output[start:end] = candidate.cpu().numpy().astype(np.complex64)
        print(f"V9 projection {end}/{len(base)}", flush=True)

    output.flush()
    finite = True
    for start in range(0, len(output), 8):
        block = np.asarray(output[start : start + 8])
        finite &= bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
    return {
        "path": PATHS.relative(OUTPUT),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "bytes": int(OUTPUT.stat().st_size),
        "relative_change_vs_v7_mean": float(np.mean(change_v7)),
        "relative_change_vs_v7_median": float(np.median(change_v7)),
        "relative_change_vs_v7_p90": float(np.quantile(change_v7, 0.90)),
        "relative_change_vs_v8_mean": float(np.mean(change_v8)),
        "relative_change_vs_v8_median": float(np.median(change_v8)),
        "relative_change_vs_v8_p90": float(np.quantile(change_v8, 0.90)),
        "power_ratio_mean": float(np.mean(power_ratio)),
        "power_ratio_max_abs_error": float(np.max(np.abs(power_ratio - 1.0))),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    required = (V7_TEST, V8_TEST, OLD_PAS, OLD_PDP, JOINT_PAS, JOINT_PDP)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    output = generate(torch.device(args.device), args.batch_size)
    payload = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_generated_not_packaged",
        "base": PATHS.relative(V7_TEST),
        "policy": asdict(POLICY),
        "targets": {
            "old_pas": PATHS.relative(OLD_PAS),
            "joint_pas": PATHS.relative(JOINT_PAS),
            "old_pdp": PATHS.relative(OLD_PDP),
            "joint_pdp": PATHS.relative(JOINT_PDP),
        },
        "absolute_sionna_phase_used": False,
        "point_power_preserved": True,
        "output": output,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
