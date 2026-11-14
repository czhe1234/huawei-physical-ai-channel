#!/usr/bin/env python3
"""Fine-tune the v439 physical model on dense independent v442 data."""

from __future__ import annotations

import json

import numpy as np
import torch

import train_v438_geometry_path_support_large_model as v438
import train_v439_component_route_long_model as trainer
from project_paths import PATHS


VERSION_NUMBER = 443
VERSION = "v443_dense_component_route_model"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
WARMSTART = (
    PATHS.checkpoints
    / "v439_component_route_long_model"
    / "component_route_best.pt"
)

v438.CACHE = CACHE
trainer.VERSION_NUMBER = VERSION_NUMBER
trainer.VERSION = VERSION
trainer.BASE_CACHE = CACHE
trainer.COMPONENT_CACHE = CACHE
trainer.CHECKPOINT_DIR = PATHS.checkpoints / VERSION
trainer.REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
trainer.REPORT = trainer.REPORT_DIR / f"{VERSION}.json"
trainer.WARMSTART = WARMSTART


def main() -> None:
    args = trainer.parse_args()
    for path in (
        CACHE / "route_raster.npy",
        CACHE / "component_features.npy",
        CACHE / "component_mass.npy",
        WARMSTART,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    trainer.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model, training, data = trainer.train(args)
    if args.smoke:
        print(json.dumps({"status": "smoke_pass", "training": training}, indent=2))
        return

    device = next(model.parameters()).device
    rng = np.random.default_rng(args.seed + 1)
    holdout = trainer.evaluate_model(
        model, data, data.holdout_ids, device, args.eval_batch_size
    )
    shuffled_ids = rng.permutation(data.holdout_ids)
    shuffled = trainer.evaluate_model(
        model,
        data,
        data.holdout_ids,
        device,
        args.eval_batch_size,
        geometry_ids=shuffled_ids,
    )
    baseline_model = trainer.ComponentRouteTransformer(args.coordinate_dropout).to(device)
    baseline_checkpoint = torch.load(WARMSTART, map_location=device, weights_only=True)
    baseline_model.load_state_dict(baseline_checkpoint["model"])
    baseline = trainer.evaluate_model(
        baseline_model,
        data,
        data.holdout_ids,
        device,
        args.eval_batch_size,
    )
    train_sample_ids = np.sort(rng.choice(data.train_ids, 10_000, replace=False))
    train_sample = trainer.evaluate_model(
        model, data, train_sample_ids, device, args.eval_batch_size
    )
    report = {
        "method": VERSION,
        "status": "independent_dense_spatial_holdout_evaluated",
        "primary_hypothesis": (
            "Dense global, wall-focused, and track-structured Sionna coverage reduces "
            "the strong nearest-training-distance error observed in v439."
        ),
        "data": {
            "source": "v442 compact Sionna dataset plus official point-cloud geometry",
            "train_count": int(len(data.train_ids)),
            "validation_count": int(len(data.validation_ids)),
            "holdout_count": int(len(data.holdout_ids)),
            "validation_holdout_min_v437_distance_m": 3.0,
        },
        "parameters": trainer.serializable_args(args),
        "single_gpu": {
            "logical_device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0),
        },
        "training": training,
        "train_sample": train_sample,
        "holdout": {
            "frozen_v439": baseline,
            "v443": holdout,
            "v443_shuffled_geometry": shuffled,
            "v443_minus_frozen_v439": {
                name: float(holdout[name] - baseline[name])
                for name in ("score", "joint_sh_cosine", "pdp_cosine")
            },
            "v443_minus_shuffled": {
                name: float(holdout[name] - shuffled[name])
                for name in ("score", "joint_sh_cosine", "pdp_cosine")
            },
        },
    }
    trainer.REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["holdout"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
