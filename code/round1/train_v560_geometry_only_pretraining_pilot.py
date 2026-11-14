#!/usr/bin/env python3
"""Train matched legacy and geometry-only v560 path-token pilot models."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import train_v438_geometry_path_support_large_model as v438
import train_v439_component_route_long_model as trainer
import train_v445_candidate_path_token_model as v445
from project_paths import PATHS


VERSION_NUMBER = 560
BASE_CACHE = PATHS.cache_dir(443)
PILOT_CACHE = PATHS.cache_dir(VERSION_NUMBER) / "pilot"
LEGACY_CACHE = PATHS.cache_dir(445)
WARMSTART = (
    PATHS.checkpoints / "v443_dense_component_route_model" / "component_route_best.pt"
)
FULL_V445 = (
    PATHS.checkpoints / "v445_candidate_path_token_model" / "component_route_best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("legacy", "clean", "clean_finetune"), required=True
    )
    parser.add_argument("--max-paths", type=int, choices=(128, 256), default=128)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--eval-batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--min-learning-rate", type=float, default=3.0e-7)
    parser.add_argument("--weight-decay", type=float, default=2.0e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--coordinate-dropout", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=56_001)
    parser.add_argument("--protect-warmstart", action="store_true", default=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.max_paths != 128:
        raise ValueError("The bitwise-controlled v560 pilot uses the retained K=128 cache")
    args.resume = None
    return args


class PilotTrainingData(trainer.TrainingData):
    path_cache = LEGACY_CACHE
    max_paths = 128
    pilot_count = 20_000
    exclude_saturated = False

    def __init__(self) -> None:
        super().__init__()
        self.path = {
            name: np.load(self.path_cache / f"path_candidate_{name}.npy", mmap_mode="r")
            for name in ("feature", "valid", "mass")
        }
        if len(self.path["feature"]) < self.pilot_count:
            raise ValueError(
                f"Path cache has {len(self.path['feature'])} rows, need {self.pilot_count}"
            )
        self.train_ids = self.train_ids[self.train_ids < self.pilot_count]
        self.validation_ids = self.validation_ids[self.validation_ids < self.pilot_count]
        self.holdout_ids = self.holdout_ids[self.holdout_ids < self.pilot_count]
        if self.exclude_saturated:
            complete = np.sum(
                np.asarray(self.path["valid"][: self.pilot_count, :128], dtype=np.int16),
                axis=1,
            ) < 128
            self.train_ids = self.train_ids[complete[self.train_ids]]
            self.validation_ids = self.validation_ids[complete[self.validation_ids]]
            self.holdout_ids = self.holdout_ids[complete[self.holdout_ids]]
        nonzero_train = self.train_ids[~self.values["zero_channel"][self.train_ids]]
        self.power_mean = float(np.mean(self.values["log_power"][nonzero_train]))
        self.power_std = float(np.std(self.values["log_power"][nonzero_train]))
        log_count = np.log1p(self.values["path_count"][self.train_ids].astype(np.float32))
        self.count_mean = float(np.mean(log_count))
        self.count_std = float(np.std(log_count))

    def batch(
        self,
        ids: np.ndarray,
        device: torch.device,
        *,
        geometry_ids: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        output = super().batch(ids, device, geometry_ids=geometry_ids)
        source_ids = ids if geometry_ids is None else geometry_ids
        output["path_candidate_feature"] = torch.from_numpy(
            np.asarray(
                self.path["feature"][source_ids, : self.max_paths], dtype=np.float32
            )
        ).to(device)
        output["path_candidate_valid"] = torch.from_numpy(
            np.asarray(self.path["valid"][source_ids, : self.max_paths], dtype=np.bool_)
        ).to(device)
        output["path_candidate_mass"] = torch.from_numpy(
            np.asarray(self.path["mass"][ids, : self.max_paths], dtype=np.float32)
        ).to(device)
        mass = output["path_candidate_mass"]
        output["path_candidate_mass"] = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return output


def warmstart_full_v445(model: torch.nn.Module) -> dict[str, object]:
    checkpoint = torch.load(FULL_V445, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("candidate_"))
    return {
        "source": PATHS.relative(FULL_V445),
        "mode": "full_v445_then_complete_set_only_finetune",
        "loaded_tensor_count": len(checkpoint["model"]),
        "loaded_parameter_count": int(
            sum(value.numel() for value in checkpoint["model"].values())
        ),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


def main() -> None:
    args = parse_args()
    manifest = np.load(PILOT_CACHE / "positions.npz")
    pilot_count = len(manifest["positions"])
    # The old cache is the only bitwise-comparable source. Rows below K=128
    # contain every returned path and therefore have no power-ranked membership.
    path_cache = LEGACY_CACHE
    for path in (
        BASE_CACHE / "route_raster.npy",
        BASE_CACHE / "component_features.npy",
        path_cache / "path_candidate_feature.npy",
        path_cache / "path_candidate_valid.npy",
        path_cache / "path_candidate_mass.npy",
        WARMSTART,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"v560 requires exactly one visible CUDA GPU, got {torch.cuda.device_count()}"
        )

    PilotTrainingData.path_cache = path_cache
    PilotTrainingData.max_paths = args.max_paths
    PilotTrainingData.pilot_count = pilot_count
    PilotTrainingData.exclude_saturated = args.variant != "legacy"
    run_name = f"{args.variant}_k{args.max_paths}"
    checkpoint_dir = (
        PATHS.checkpoints / "v560_geometry_only_pretraining_pilot" / run_name
    )
    report_dir = PATHS.version_report_dir(VERSION_NUMBER)
    report_path = report_dir / f"v560_pretraining_{run_name}.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    trainer.VERSION_NUMBER = VERSION_NUMBER
    trainer.VERSION = f"v560_pretraining_{run_name}"
    trainer.BASE_CACHE = BASE_CACHE
    trainer.COMPONENT_CACHE = BASE_CACHE
    trainer.CHECKPOINT_DIR = checkpoint_dir
    trainer.REPORT_DIR = report_dir
    trainer.REPORT = report_path
    trainer.WARMSTART = WARMSTART
    trainer.TrainingData = PilotTrainingData
    trainer.ComponentRouteTransformer = v445.CandidatePathTokenModel
    trainer.warmstart = (
        warmstart_full_v445
        if args.variant == "clean_finetune"
        else v445.warmstart_and_freeze
    )
    trainer.model_loss = v445.candidate_model_loss
    v438.CACHE = BASE_CACHE

    started = time.time()
    model, training, data = trainer.train(args)
    if args.smoke:
        print(json.dumps({"status": "smoke_pass", "training": training}, indent=2))
        return
    device = next(model.parameters()).device
    rng = np.random.default_rng(args.seed + 1)
    holdout = trainer.evaluate_model(
        model, data, data.holdout_ids, device, args.eval_batch_size
    )
    shuffled = trainer.evaluate_model(
        model,
        data,
        data.holdout_ids,
        device,
        args.eval_batch_size,
        geometry_ids=rng.permutation(data.holdout_ids),
    )
    baseline_model = v445.BaseComponentModel(args.coordinate_dropout).to(device)
    baseline_checkpoint = torch.load(WARMSTART, map_location=device, weights_only=True)
    baseline_model.load_state_dict(baseline_checkpoint["model"])
    baseline = trainer.evaluate_model(
        baseline_model, data, data.holdout_ids, device, args.eval_batch_size
    )
    report = {
        "method": f"v560_pretraining_{run_name}",
        "status": "synthetic_pilot_holdout_evaluated",
        "primary_hypothesis": (
            "A path-token branch trained with power-independent candidate membership "
            "learns transferable path existence and energy rather than a top-K power cue."
        ),
        "variant": args.variant,
        "candidate_membership": (
            "complete_returned_set_only_path_count_below_128"
            if args.variant != "legacy"
            else "legacy_including_power_selected_top128_rows"
        ),
        "confounding_control": (
            "Both variants read the same v445 token and target arrays. Clean differs only "
            "by excluding rows saturated at K=128; no Sionna rerun enters training."
        ),
        "pilot_count": pilot_count,
        "split_counts": {
            "train": int(len(data.train_ids)),
            "validation": int(len(data.validation_ids)),
            "holdout": int(len(data.holdout_ids)),
        },
        "max_paths": args.max_paths,
        "parameters": trainer.serializable_args(args),
        "single_gpu": {
            "logical_device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0),
        },
        "training": training,
        "holdout": {
            "v443": baseline,
            run_name: holdout,
            f"{run_name}_shuffled_all_geometry": shuffled,
            "minus_v443": {
                key: float(holdout[key] - baseline[key])
                for key in ("score", "joint_sh_cosine", "pdp_cosine")
            },
            "minus_shuffled": {
                key: float(holdout[key] - shuffled[key])
                for key in ("score", "joint_sh_cosine", "pdp_cosine")
            },
        },
        "seconds": float(time.time() - started),
        "outputs": {
            "checkpoint": training["checkpoint"],
            "report": PATHS.relative(report_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["holdout"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
