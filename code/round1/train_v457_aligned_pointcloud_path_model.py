#!/usr/bin/env python3
"""Train a synthetic path branch with deployment-aligned point-cloud tokens."""

from __future__ import annotations

import json

import numpy as np
import torch

import train_v439_component_route_long_model as trainer
import train_v445_candidate_path_token_model as v445
from project_paths import PATHS


VERSION_NUMBER = 457
VERSION = "v457_aligned_pointcloud_path_model"
TOKEN_CACHE = PATHS.cache_dir(VERSION_NUMBER)
BASE_CACHE = PATHS.cache_dir(443)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
WARMSTART = PATHS.checkpoints / "v443_dense_component_route_model" / "component_route_best.pt"


class AlignedTrainingData(trainer.TrainingData):
    def __init__(self) -> None:
        super().__init__()
        self.source_ids = np.load(TOKEN_CACHE / "source_ids.npy").astype(np.int64)
        self.path_feature = np.load(
            TOKEN_CACHE / "path_candidate_feature.npy", mmap_mode="r"
        )
        self.path_valid = np.load(
            TOKEN_CACHE / "path_candidate_valid.npy", mmap_mode="r"
        )
        self.lookup = np.full(len(self.split), -1, dtype=np.int64)
        self.lookup[self.source_ids] = np.arange(len(self.source_ids), dtype=np.int64)
        self.train_ids = self.source_ids[self.split[self.source_ids] == 0]
        self.validation_ids = self.source_ids[self.split[self.source_ids] == 1]
        self.holdout_ids = self.source_ids[self.split[self.source_ids] == 2]

    def batch(
        self,
        ids: np.ndarray,
        device: torch.device,
        *,
        geometry_ids: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        output = super().batch(ids, device, geometry_ids=geometry_ids)
        source_ids = ids if geometry_ids is None else geometry_ids
        local = self.lookup[source_ids]
        if np.any(local < 0):
            raise IndexError("Requested geometry row outside the aligned token manifest")
        output["path_candidate_feature"] = torch.from_numpy(
            np.asarray(self.path_feature[local], dtype=np.float32)
        ).to(device)
        output["path_candidate_valid"] = torch.from_numpy(
            np.asarray(self.path_valid[local], dtype=np.bool_)
        ).to(device)
        return output


def aligned_loss(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    return v445.base_model_loss(output, batch)


def evaluate_base(
    data: AlignedTrainingData,
    ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    checkpoint = torch.load(WARMSTART, map_location="cpu", weights_only=True)
    model = v445.BaseComponentModel(0.0)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    return trainer.evaluate_model(model, data, ids, device, batch_size)


def main() -> None:
    args = trainer.parse_args()
    for path in (
        TOKEN_CACHE / "source_ids.npy",
        TOKEN_CACHE / "path_candidate_feature.npy",
        TOKEN_CACHE / "path_candidate_valid.npy",
        WARMSTART,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.VERSION_NUMBER = VERSION_NUMBER
    trainer.VERSION = VERSION
    trainer.BASE_CACHE = BASE_CACHE
    trainer.COMPONENT_CACHE = BASE_CACHE
    trainer.CHECKPOINT_DIR = CHECKPOINT_DIR
    trainer.REPORT_DIR = REPORT_DIR
    trainer.REPORT = REPORT
    trainer.WARMSTART = WARMSTART
    trainer.TrainingData = AlignedTrainingData
    trainer.ComponentRouteTransformer = v445.CandidatePathTokenModel
    trainer.warmstart = v445.warmstart_and_freeze
    trainer.model_loss = aligned_loss
    model, training, data = trainer.train(args)
    if args.smoke:
        print(json.dumps({"status": "smoke_pass", "training": training}, indent=2))
        return
    device = next(model.parameters()).device
    baseline = evaluate_base(data, data.holdout_ids, device, args.eval_batch_size)
    aligned = trainer.evaluate_model(
        model, data, data.holdout_ids, device, args.eval_batch_size
    )
    rng = np.random.default_rng(args.seed + 1)
    shuffled_ids = rng.permutation(data.holdout_ids)
    shuffled = trainer.evaluate_model(
        model,
        data,
        data.holdout_ids,
        device,
        args.eval_batch_size,
        geometry_ids=shuffled_ids,
    )
    delta = {
        name: float(aligned[name] - baseline[name])
        for name in ("score", "joint_sh_cosine", "pdp_cosine")
    }
    report = {
        "method": VERSION,
        "status": "pilot_holdout_evaluated",
        "primary_hypothesis": (
            "A path encoder trained on official-point-cloud candidates at synthetic "
            "positions transfers because train and deployment candidate membership are identical."
        ),
        "parameters": trainer.serializable_args(args),
        "data": {
            "train_count": int(len(data.train_ids)),
            "validation_count": int(len(data.validation_ids)),
            "holdout_count": int(len(data.holdout_ids)),
        },
        "training": training,
        "holdout": {
            "v443": baseline,
            "v457": aligned,
            "v457_shuffled_all_geometry": shuffled,
            "v457_minus_v443": delta,
        },
        "decision": {
            "criterion": "holdout score gain >= 0.01 with both joint S-H and PDP positive",
            "passed": bool(
                delta["score"] >= 0.01
                and delta["joint_sh_cosine"] > 0
                and delta["pdp_cosine"] > 0
            ),
        },
        "compliance_boundary": {
            "uses": [
                "self-generated Sionna labels at auxiliary random positions",
                "official public point cloud and coordinates for geometry-only inputs",
                "AI encoder with no path power or complex coefficient input",
            ],
            "does_not_use": [
                "Sionna CFR or path coefficients at official test coordinates",
                "official test H",
                "v212/v426/v436/v449 teacher predictions",
                "online score for model fitting",
            ],
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["holdout"], indent=2, ensure_ascii=False), flush=True)
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
