#!/usr/bin/env python3
"""Train a residual joint-spectrum calibrator over frozen v439 predictions."""

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

import train_v438_geometry_path_support_large_model as v438
from project_paths import PATHS


VERSION_NUMBER = 440
VERSION = "v440_component_spectral_calibrator"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
TARGET_CACHE = PATHS.cache_dir(438)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CHECKPOINT = CHECKPOINT_DIR / "spectral_calibrator_best.pt"

S_BINS = 192
H_BINS = 16
COMPONENTS = 82


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--ema-decay", type=float, default=0.998)
    parser.add_argument("--seed", type=int, default=44_001)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class TrainingData:
    def __init__(self) -> None:
        self.base_joint = np.load(CACHE / "base_joint.npy", mmap_mode="r")
        self.path_delay = np.load(CACHE / "path_delay.npy", mmap_mode="r")
        self.component_delay = np.load(CACHE / "component_delay.npy", mmap_mode="r")
        self.component_h = np.load(CACHE / "component_h.npy", mmap_mode="r")
        self.component_mass = np.load(CACHE / "component_mass.npy", mmap_mode="r")
        self.target_joint = np.load(TARGET_CACHE / "joint_sh.npy", mmap_mode="r")
        self.target_pdp = np.load(TARGET_CACHE / "pdp.npy", mmap_mode="r")
        self.zero = np.load(TARGET_CACHE / "zero_channel.npy", mmap_mode="r")
        self.split = np.load(TARGET_CACHE / "split.npy", mmap_mode="r")
        self.train_ids = np.flatnonzero(self.split == 0).astype(np.int64)
        self.validation_ids = np.flatnonzero(self.split == 1).astype(np.int64)
        self.holdout_ids = np.flatnonzero(self.split == 2).astype(np.int64)

    def batch(
        self,
        ids: np.ndarray,
        device: torch.device,
        *,
        auxiliary_ids: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        source = ids if auxiliary_ids is None else auxiliary_ids
        output = {
            "base_joint": torch.from_numpy(
                np.asarray(self.base_joint[ids], dtype=np.float32)
            ).to(device),
            "path_delay": torch.from_numpy(
                np.asarray(self.path_delay[source], dtype=np.float32)
            ).to(device),
            "component_delay": torch.from_numpy(
                np.asarray(self.component_delay[source], dtype=np.float32)
            ).to(device),
            "component_h": torch.from_numpy(
                np.asarray(self.component_h[source], dtype=np.float32)
            ).to(device),
            "component_mass": torch.from_numpy(
                np.asarray(self.component_mass[source], dtype=np.float32)
            ).to(device),
            "target_joint": torch.from_numpy(
                np.asarray(self.target_joint[ids], dtype=np.float32)
            ).to(device),
            "target_pdp": torch.from_numpy(
                np.asarray(self.target_pdp[ids], dtype=np.float32)
            ).to(device),
            "nonzero": torch.from_numpy(~np.asarray(self.zero[ids], dtype=bool)).to(device),
        }
        return output


class FilmResidualBlock(nn.Module):
    def __init__(self, channels: int, context_dim: int, dilation: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(16, channels)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.norm2 = nn.GroupNorm(16, channels)
        self.conv2 = nn.Conv2d(
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
        hidden = hidden * (1.0 + 0.10 * scale[:, :, None, None])
        hidden = hidden + 0.10 * bias[:, :, None, None]
        hidden = self.conv1(F.gelu(hidden))
        hidden = self.conv2(F.gelu(self.norm2(hidden)))
        return value + 0.5 * hidden


class ComponentSpectralCalibrator(nn.Module):
    def __init__(self, channels: int = 128, blocks: int = 16) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(COMPONENTS, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.stem = nn.Conv2d(8, channels, 3, padding=1, padding_mode="circular")
        dilation = [1, 2, 4, 8] * (blocks // 4)
        self.blocks = nn.ModuleList(
            FilmResidualBlock(channels, 256, value) for value in dilation
        )
        self.head = nn.Sequential(
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def s_aligned_delay(value: torch.Tensor) -> torch.Tensor:
        reverse = torch.cat(
            [
                torch.zeros(1, device=value.device, dtype=torch.long),
                torch.arange(S_BINS - 1, 0, -1, device=value.device),
            ]
        )
        return value[:, reverse]

    def forward(
        self,
        base_joint: torch.Tensor,
        path_delay: torch.Tensor,
        component_delay: torch.Tensor,
        component_h: torch.Tensor,
        component_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        epsilon = 1e-8
        base_log = torch.log(torch.clamp(base_joint, min=epsilon))
        path_s = self.s_aligned_delay(path_delay)
        component_full = torch.zeros(
            len(base_joint), S_BINS, device=base_joint.device, dtype=base_joint.dtype
        )
        component_full[:, :32] = component_delay
        component_s = self.s_aligned_delay(component_full)
        s_coordinate = torch.linspace(
            0.0, 2.0 * math.pi, S_BINS, device=base_joint.device
        )
        h_coordinate = torch.linspace(
            0.0, 2.0 * math.pi, H_BINS, device=base_joint.device
        )
        shape = base_joint.shape
        feature = torch.stack(
            [
                torch.clamp(base_log, min=-16.0),
                torch.log(torch.clamp(path_s, min=epsilon))[:, :, None].expand(shape),
                torch.log(torch.clamp(component_s, min=epsilon))[:, :, None].expand(shape),
                torch.log(torch.clamp(component_h, min=epsilon))[:, None, :].expand(shape),
                torch.sin(s_coordinate)[None, :, None].expand(shape),
                torch.cos(s_coordinate)[None, :, None].expand(shape),
                torch.sin(h_coordinate)[None, None, :].expand(shape),
                torch.cos(h_coordinate)[None, None, :].expand(shape),
            ],
            dim=1,
        )
        context = self.context(component_mass)
        hidden = self.stem(feature)
        for block in self.blocks:
            hidden = block(hidden, context)
        correction = self.head(hidden).squeeze(1)
        logits = base_log + correction
        probability = F.softmax(logits.flatten(1), dim=1).reshape_as(base_joint)
        return probability, correction


def cosine_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    pred = prediction.flatten(1)
    truth = target.flatten(1)
    cosine = torch.sum(pred * truth, dim=1) / torch.clamp(
        torch.linalg.vector_norm(pred, dim=1)
        * torch.linalg.vector_norm(truth, dim=1),
        min=1e-8,
    )
    return torch.mean(1.0 - cosine[mask])


def loss_function(
    prediction: torch.Tensor,
    correction: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["nonzero"]
    flat = torch.clamp(prediction.flatten(1), min=1e-12)
    target = batch["target_joint"].flatten(1)
    cross_entropy = torch.mean(-torch.sum(target[mask] * torch.log(flat[mask]), dim=1))
    joint = cosine_loss(prediction, batch["target_joint"], mask)
    reverse = torch.cat(
        [
            torch.zeros(1, device=prediction.device, dtype=torch.long),
            torch.arange(S_BINS - 1, 0, -1, device=prediction.device),
        ]
    )
    pdp = torch.sum(prediction[:, reverse], dim=2)
    pdp_loss = cosine_loss(pdp, batch["target_pdp"], mask)
    regularization = torch.mean(correction[mask] ** 2)
    total = cross_entropy + 1.20 * joint + 0.80 * pdp_loss + 1e-4 * regularization
    return total, {
        "cross_entropy": float(cross_entropy.detach()),
        "joint_cosine_loss": float(joint.detach()),
        "pdp_cosine_loss": float(pdp_loss.detach()),
        "correction_l2": float(regularization.detach()),
    }


def cosine_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return v438.cosine_rows(prediction, target)


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
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        source = None if auxiliary_ids is None else auxiliary_ids[start:end]
        batch = data.batch(ids[start:end], device, auxiliary_ids=source)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, _ = model(
                batch["base_joint"],
                batch["path_delay"],
                batch["component_delay"],
                batch["component_h"],
                batch["component_mass"],
            )
        joint_parts.append(prediction.float().cpu().numpy())
    joint = np.concatenate(joint_parts)
    zero = np.asarray(data.zero[ids], dtype=bool)
    nonzero = ~zero
    target_joint = np.asarray(data.target_joint[ids], dtype=np.float32)
    target_pdp = np.asarray(data.target_pdp[ids], dtype=np.float32)
    reverse = np.concatenate([[0], np.arange(S_BINS - 1, 0, -1)])
    pdp = np.sum(joint[:, reverse], axis=2)
    joint_cosine = float(np.mean(cosine_rows(joint[nonzero], target_joint[nonzero])))
    pdp_cosine = float(np.mean(cosine_rows(pdp[nonzero], target_pdp[nonzero])))
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
            f"v440 requires exactly one visible GPU, got {torch.cuda.device_count()}"
        )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    data = TrainingData()
    model = ComponentSpectralCalibrator().to(device)
    ema = copy.deepcopy(model).to(device).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_ids = data.train_ids[:2048] if args.smoke else data.train_ids
    validation_ids = data.validation_ids[:512] if args.smoke else data.validation_ids
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
        f"v440 epoch=0 baseline={initial_validation['score']:.6f} "
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
                prediction, correction = model(
                    batch["base_joint"],
                    batch["path_delay"],
                    batch["component_delay"],
                    batch["component_h"],
                    batch["component_mass"],
                )
                loss, parts = loss_function(prediction, correction, batch)
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
            f"v440 epoch={epoch} loss={row['train_loss']:.5f} "
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
    for path in (
        CACHE / "base_joint.npy",
        CACHE / "component_delay.npy",
        TARGET_CACHE / "joint_sh.npy",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
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
            "A metric-aligned residual calibrator can convert v439 component/path "
            "predictions into joint-support corrections while preserving its geometry backbone."
        ),
        "parameters": vars(args),
        "single_gpu": {
            "logical_device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0),
        },
        "training": training,
        "holdout": {
            "v439": baseline,
            "v440": holdout,
            "v440_shuffled_auxiliary": shuffled,
            "v440_minus_v439": {
                name: float(holdout[name] - baseline[name])
                for name in ("score", "joint_sh_cosine", "pdp_cosine")
            },
            "v440_minus_shuffled_auxiliary": {
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
