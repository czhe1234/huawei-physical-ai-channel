#!/usr/bin/env python3
"""Build full-4000 cross-fitted E88 endpoints for reliability-gate training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time

import numpy as np
import torch

from project_paths import PATHS
import train_e88_dense_path_residual_student as e88
from train_full_path_pdp_link_gate import FOLD_IDS
import train_surface_interaction_energy_model as e85


VERSION = "Round2-E91-E88-full4000-OOF-endpoints"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e88_full4000_oof_endpoints"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e88_full4000_oof_endpoints"
OOF_ENDPOINTS = ARTIFACT_DIR / "full4000_oof_e88_endpoints.npz"
REPORT = ANALYSIS_DIR / "full4000_oof_e88_endpoints.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    config = e88.Config(
        finetune_epochs=args.epochs,
        finetune_batch_size=args.batch_size,
    )
    geometry = e88.GeometryStore("train", device)
    fold_id = np.load(FOLD_IDS).astype(np.int8)
    if fold_id.shape != (len(geometry.positions),) or np.any(fold_id < 0):
        raise ValueError("full-4000 fold assignment is invalid")
    teacher_cache = np.load(
        PATHS.artifacts
        / "round2"
        / "offline_sionna_geometry_student"
        / "raw_sionna_sufficient_teacher.npz",
        allow_pickle=False,
    )
    official_pas = torch.as_tensor(
        np.array(np.load(e85.TARGET_PAS, mmap_mode="r"), dtype=np.float32, copy=True),
        device=device,
    )
    official_pdp = torch.as_tensor(
        np.array(np.load(e85.TARGET_PDP, mmap_mode="r"), dtype=np.float32, copy=True),
        device=device,
    )
    teacher_pas = torch.as_tensor(teacher_cache["pas"].astype(np.float32), device=device)
    teacher_pdp = torch.as_tensor(teacher_cache["pdp"].astype(np.float32), device=device)
    power = np.asarray(np.load(e85.POINT_POWER), dtype=np.float64)
    count = len(geometry.positions)
    outputs = {
        "final_pas": np.zeros((count, 2, 16, 8, 4), dtype=np.float16),
        "final_pdp": np.zeros((count, 2, 4, 192), dtype=np.float16),
        "path_pas": np.zeros((count, 2, 16, 8, 4), dtype=np.float16),
        "path_pdp": np.zeros((count, 2, 4, 192), dtype=np.float16),
        "confidence": np.zeros((count, 2), dtype=np.float16),
    }
    training = []
    started = time.time()
    for outer in range(5):
        validation_fold = (outer + 1) % 5
        held = np.flatnonzero(fold_id == outer)
        validation = np.flatnonzero(fold_id == validation_fold)
        fit = np.flatnonzero((fold_id != outer) & (fold_id != validation_fold))
        model, record = e88.train_official_fold(
            outer=outer,
            fit=fit,
            validation=validation,
            geometry=geometry,
            official_pas=official_pas,
            official_pdp=official_pdp,
            teacher_pas=teacher_pas,
            teacher_pdp=teacher_pdp,
            power=power,
            config=config,
        )
        prediction = e88.predict_official(
            model, geometry, held, config.finetune_batch_size
        )
        outputs["final_pas"][held] = prediction["pas"].astype(np.float16)
        outputs["final_pdp"][held] = prediction["pdp"].astype(np.float16)
        outputs["path_pas"][held] = prediction["path_pas"].astype(np.float16)
        outputs["path_pdp"][held] = prediction["path_pdp"].astype(np.float16)
        outputs["confidence"][held, 0] = prediction["valid"].astype(np.float16)
        outputs["confidence"][held, 1] = prediction["reliability"].astype(np.float16)
        checkpoint_path = ARTIFACT_DIR / f"outer{outer}_full4000_student.pt"
        torch.save(
            {
                "experiment": VERSION,
                "outer": outer,
                "config": asdict(config),
                "state_dict": model.state_dict(),
                "sionna_inputs_required_for_inference": False,
            },
            checkpoint_path,
        )
        training.append(
            {
                **record,
                "held_count": int(len(held)),
                "checkpoint": PATHS.relative(checkpoint_path),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    np.savez_compressed(
        OOF_ENDPOINTS,
        fold_id=fold_id,
        **outputs,
    )
    endpoint = e88.endpoint_summary(
        outputs["final_pas"].astype(np.float32),
        outputs["final_pdp"].astype(np.float32),
        official_pas,
        official_pdp,
        fold_id,
        device,
    )
    path_endpoint = e88.endpoint_summary(
        outputs["path_pas"].astype(np.float32),
        outputs["path_pdp"].astype(np.float32),
        official_pas,
        official_pdp,
        fold_id,
        device,
    )
    report = {
        "experiment": VERSION,
        "status": "strict_complete",
        "configuration": asdict(config),
        "fold_counts": np.bincount(fold_id, minlength=5).astype(int).tolist(),
        "endpoint": endpoint,
        "path_endpoint": path_endpoint,
        "training": training,
        "output": PATHS.relative(OOF_ENDPOINTS),
        "seconds": time.time() - started,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
