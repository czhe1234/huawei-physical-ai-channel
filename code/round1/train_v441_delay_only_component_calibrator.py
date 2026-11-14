#!/usr/bin/env python3
"""Train a delay-only residual calibrator over frozen v439 predictions."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_v440_component_spectral_calibrator as v440
from project_paths import PATHS


VERSION_NUMBER = 441
VERSION = "v441_delay_only_component_calibrator"
CACHE = PATHS.cache_dir(440)
TARGET_CACHE = PATHS.cache_dir(438)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CHECKPOINT = CHECKPOINT_DIR / "delay_calibrator_best.pt"

DELAY_BINS = 192
COMPONENTS = 82


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--seed", type=int, default=44_101)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class TrainingData(v440.TrainingData):
    pass


def reverse_index(device: torch.device) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.long),
            torch.arange(DELAY_BINS - 1, 0, -1, device=device),
        ]
    )


class DelayResidualBlock(nn.Module):
    def __init__(self, channels: int, context_dim: int, dilation: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(16, channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.norm2 = nn.GroupNorm(16, channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.film = nn.Linear(context_dim, 2 * channels)

    def forward(self, value: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale, bias = self.film(context).chunk(2, dim=1)
        hidden = self.norm1(value)
        hidden = hidden * (1.0 + 0.10 * scale[:, :, None])
        hidden = hidden + 0.10 * bias[:, :, None]
        hidden = self.conv1(F.gelu(hidden))
        hidden = self.conv2(F.gelu(self.norm2(hidden)))
        return value + 0.5 * hidden


class DelayOnlyCalibrator(nn.Module):
    def __init__(self, channels: int = 256, blocks: int = 16) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(COMPONENTS, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
        )
        self.stem = nn.Conv1d(5, channels, 5, padding=2, padding_mode="circular")
        dilation = [1, 2, 4, 8] * (blocks // 4)
        self.blocks = nn.ModuleList(
            DelayResidualBlock(channels, 384, value) for value in dilation
        )
        self.head = nn.Sequential(
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Conv1d(channels, 1, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        base_joint: torch.Tensor,
        path_delay: torch.Tensor,
        component_delay: torch.Tensor,
        component_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reverse = reverse_index(base_joint.device)
        base_pdp = torch.sum(base_joint[:, reverse], dim=2)
        component_full = torch.zeros_like(base_pdp)
        component_full[:, :32] = component_delay
        coordinate = torch.linspace(
            0.0, 2.0 * math.pi, DELAY_BINS, device=base_joint.device
        )
        epsilon = 1e-8
        feature = torch.stack(
            [
                torch.clamp(torch.log(torch.clamp(base_pdp, min=epsilon)), min=-16.0),
                torch.clamp(torch.log(torch.clamp(path_delay, min=epsilon)), min=-16.0),
                torch.clamp(torch.log(torch.clamp(component_full, min=epsilon)), min=-16.0),
                torch.sin(coordinate)[None, :].expand_as(base_pdp),
                torch.cos(coordinate)[None, :].expand_as(base_pdp),
            ],
            dim=1,
        )
        context = self.context(component_mass)
        hidden = self.stem(feature)
        for block in self.blocks:
            hidden = block(hidden, context)
        correction = self.head(hidden).squeeze(1)
        corrected_pdp = F.softmax(
            torch.log(torch.clamp(base_pdp, min=epsilon)) + correction,
            dim=1,
        )
        base_s_mass = torch.sum(base_joint, dim=2)
        conditional_h = base_joint / torch.clamp(base_s_mass[:, :, None], min=epsilon)
        corrected_s_mass = corrected_pdp[:, reverse]
        corrected_joint = conditional_h * corrected_s_mass[:, :, None]
        return corrected_joint, corrected_pdp, correction


def cosine_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    prediction = prediction.flatten(1)
    target = target.flatten(1)
    cosine = torch.sum(prediction * target, dim=1) / torch.clamp(
        torch.linalg.vector_norm(prediction, dim=1)
        * torch.linalg.vector_norm(target, dim=1),
        min=1e-8,
    )
    return torch.mean(1.0 - cosine[mask])


def loss_function(
    joint: torch.Tensor,
    pdp: torch.Tensor,
    correction: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["nonzero"]
    cross_entropy = torch.mean(
        -torch.sum(
            batch["target_pdp"][mask]
            * torch.log(torch.clamp(pdp[mask], min=1e-12)),
            dim=1,
        )
    )
    pdp_cosine = cosine_loss(pdp, batch["target_pdp"], mask)
    joint_cosine = cosine_loss(joint, batch["target_joint"], mask)
    regularization = torch.mean(correction[mask] ** 2)
    total = (
        cross_entropy
        + 2.0 * pdp_cosine
        + 0.50 * joint_cosine
        + 1e-4 * regularization
    )
    return total, {
        "cross_entropy": float(cross_entropy.detach()),
        "pdp_cosine_loss": float(pdp_cosine.detach()),
        "joint_cosine_loss": float(joint_cosine.detach()),
        "correction_l2": float(regularization.detach()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: TrainingData,
    ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    auxiliary_ids: np.ndarray | None = None,
) -> dict[str, float]:
    model.eval()
    joint_parts = []
    pdp_parts = []
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        source = None if auxiliary_ids is None else auxiliary_ids[start:end]
        batch = data.batch(ids[start:end], device, auxiliary_ids=source)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            joint, pdp, _ = model(
                batch["base_joint"],
                batch["path_delay"],
                batch["component_delay"],
                batch["component_mass"],
            )
        joint_parts.append(joint.float().cpu().numpy())
        pdp_parts.append(pdp.float().cpu().numpy())
    joint = np.concatenate(joint_parts)
    pdp = np.concatenate(pdp_parts)
    zero = np.asarray(data.zero[ids], dtype=bool)
    nonzero = ~zero
    target_joint = np.asarray(data.target_joint[ids], dtype=np.float32)
    target_pdp = np.asarray(data.target_pdp[ids], dtype=np.float32)
    joint_cosine = float(
        np.mean(v440.cosine_rows(joint[nonzero], target_joint[nonzero]))
    )
    pdp_cosine = float(
        np.mean(v440.cosine_rows(pdp[nonzero], target_pdp[nonzero]))
    )
    return {
        "score": 0.60 * joint_cosine + 0.40 * pdp_cosine,
        "joint_sh_cosine": joint_cosine,
        "pdp_cosine": pdp_cosine,
        "count": int(len(ids)),
        "nonzero_count": int(np.sum(nonzero)),
    }


def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for target, source in zip(ema.parameters(), model.parameters(), strict=True):
            target.lerp_(source, 1.0 - decay)
        for target, source in zip(ema.buffers(), model.buffers(), strict=True):
            target.copy_(source)


def train(args: argparse.Namespace):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"v441 requires exactly one visible GPU, got {torch.cuda.device_count()}"
        )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    data = TrainingData()
    model = DelayOnlyCalibrator().to(device)
    ema = copy.deepcopy(model).to(device).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_ids = data.train_ids[:3072] if args.smoke else data.train_ids
    validation_ids = data.validation_ids[:768] if args.smoke else data.validation_ids
    epochs = 2 if args.smoke else args.epochs
    steps_per_epoch = math.ceil(len(train_ids) / args.batch_size)
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    min_scale = args.min_learning_rate / args.learning_rate
    initial_validation = evaluate(
        model, data, validation_ids, device, args.eval_batch_size
    )
    best_score = float(initial_validation["score"])
    best_epoch = 0
    best_source = "initial"
    stale = 0
    global_step = 0
    history = []
    started = time.time()
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": 0,
            "source": "initial",
            "validation": initial_validation,
            "args": vars(args),
        },
        CHECKPOINT,
    )
    print(
        f"v441 epoch=0 baseline={initial_validation['score']:.6f} "
        f"joint={initial_validation['joint_sh_cosine']:.6f} "
        f"pdp={initial_validation['pdp_cosine']:.6f}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_ids)
        total_loss = 0.0
        seen = 0
        part_sum: dict[str, float] = {}
        epoch_started = time.time()
        for start in range(0, len(order), args.batch_size):
            ids = order[start : start + args.batch_size]
            if global_step < warmup_steps:
                scale = max((global_step + 1) / max(warmup_steps, 1), 1e-3)
            else:
                progress = (global_step - warmup_steps) / max(
                    total_steps - warmup_steps, 1
                )
                scale = min_scale + (1.0 - min_scale) * 0.5 * (
                    1.0 + math.cos(math.pi * min(progress, 1.0))
                )
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * scale
            batch = data.batch(ids, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                joint, pdp, correction = model(
                    batch["base_joint"],
                    batch["path_delay"],
                    batch["component_delay"],
                    batch["component_mass"],
                )
                loss, parts = loss_function(joint, pdp, correction, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            update_ema(ema, model, args.ema_decay)
            global_step += 1
            total_loss += float(loss.detach()) * len(ids)
            seen += len(ids)
            for name, value in parts.items():
                part_sum[name] = part_sum.get(name, 0.0) + value * len(ids)
        raw_validation = evaluate(
            model, data, validation_ids, device, args.eval_batch_size
        )
        ema_validation = evaluate(
            ema, data, validation_ids, device, args.eval_batch_size
        )
        source, validation, selected = max(
            (("raw", raw_validation, model), ("ema", ema_validation, ema)),
            key=lambda row: row[1]["score"],
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "train_parts": {name: value / seen for name, value in part_sum.items()},
            "raw_validation": raw_validation,
            "ema_validation": ema_validation,
            "selected_source": source,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": float(time.time() - epoch_started),
        }
        history.append(row)
        print(
            f"v441 epoch={epoch} loss={row['train_loss']:.5f} "
            f"raw={raw_validation['score']:.6f} ema={ema_validation['score']:.6f} "
            f"joint={validation['joint_sh_cosine']:.6f} "
            f"pdp={validation['pdp_cosine']:.6f} "
            f"lr={row['learning_rate']:.3e} seconds={row['seconds']:.1f}",
            flush=True,
        )
        if validation["score"] > best_score + 1e-5:
            best_score = validation["score"]
            best_epoch = epoch
            best_source = source
            stale = 0
            torch.save(
                {
                    "model": selected.state_dict(),
                    "epoch": epoch,
                    "source": source,
                    "validation": validation,
                    "args": vars(args),
                },
                CHECKPOINT,
            )
        else:
            stale += 1
        if not args.smoke and epoch >= args.min_epochs and stale >= args.patience:
            break
    best = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
    return model, data, {
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "best_epoch": best_epoch,
        "best_source": best_source,
        "best_validation": best["validation"],
        "seconds": float(time.time() - started),
        "checkpoint": PATHS.relative(CHECKPOINT),
        "history": history,
    }


def main() -> None:
    args = parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model, data, training = train(args)
    if args.smoke:
        print(json.dumps({"status": "smoke_pass", "training": training}, indent=2))
        return
    device = next(model.parameters()).device
    rng = np.random.default_rng(args.seed + 1)
    permutation = rng.permutation(data.holdout_ids)
    holdout = evaluate(
        model, data, data.holdout_ids, device, args.eval_batch_size
    )
    shuffled = evaluate(
        model,
        data,
        data.holdout_ids,
        device,
        args.eval_batch_size,
        auxiliary_ids=permutation,
    )
    v439_report = json.loads(
        (PATHS.version_report_dir(439) / "v439_component_route_long_model.json").read_text(
            encoding="utf-8"
        )
    )
    baseline = v439_report["holdout"]["v439"]
    report = {
        "method": VERSION,
        "status": "synthetic_spatial_holdout_evaluated",
        "primary_hypothesis": (
            "A delay-only physical correction can exploit component/path delay predictions "
            "without diverting capacity into or damaging v439's horizontal-angle conditional."
        ),
        "parameters": vars(args),
        "single_gpu": {
            "logical_device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0),
        },
        "training": training,
        "holdout": {
            "v439": baseline,
            "v441": holdout,
            "v441_shuffled_auxiliary": shuffled,
            "v441_minus_v439": {
                name: float(holdout[name] - baseline[name])
                for name in ("score", "joint_sh_cosine", "pdp_cosine")
            },
            "v441_minus_shuffled_auxiliary": {
                name: float(holdout[name] - shuffled[name])
                for name in ("score", "joint_sh_cosine", "pdp_cosine")
            },
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["holdout"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
