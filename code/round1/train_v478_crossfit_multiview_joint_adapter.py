#!/usr/bin/env python3
"""Cross-fit an official joint S-H adapter over aligned pretrained path views."""

from __future__ import annotations

import argparse
import copy
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import analyze_v477_aligned_pretrained_path_transfer as v477
import train_v469_neural_sionna_calibrator as v469
from project_paths import PATHS


VERSION_NUMBER = 478
VERSION = "v478_crossfit_multiview_joint_adapter"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF = CACHE / "v478_oof_joint.npy"
OOF_ABLATION = CACHE / "v478_oof_joint_path_ablation.npy"
TEST_TARGET = CACHE / "v478_test_joint_target.npy"
BLENDED_TARGET = CACHE / "v478_blended_joint_target.npy"
OUTPUT = CACHE / "v478_Round1_Test_Channel.npy"
FLOOR = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=478_001)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class MultiViewAdapter(nn.Module):
    def __init__(self, point_dim: int, width: int = 96, blocks: int = 10) -> None:
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(point_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        self.input = v469.CircularConv2d(7, width)
        self.blocks = nn.ModuleList(
            [v469.JointBlock(width, 256) for _ in range(blocks)]
        )
        self.output = v469.CircularConv2d(width, 1)
        nn.init.zeros_(self.output.conv.weight)
        nn.init.zeros_(self.output.conv.bias)

    def forward(
        self,
        raw: torch.Tensor,
        base: torch.Tensor,
        quality: torch.Tensor,
        v443: torch.Tensor,
        v445: torch.Tensor,
        point: torch.Tensor,
    ) -> torch.Tensor:
        raw_log = torch.log(raw.clamp_min(FLOOR))
        base_log = torch.log(base.clamp_min(FLOOR))
        v443_log = torch.log(v443.clamp_min(FLOOR))
        v445_log = torch.log(v445.clamp_min(FLOOR))
        value = torch.stack(
            [
                raw_log / 8.0,
                base_log / 8.0,
                v443_log / 8.0,
                v445_log / 8.0,
                (raw_log - base_log) / 4.0,
                (v445_log - v443_log) / 4.0,
                quality,
            ],
            dim=1,
        )
        condition = self.point(point)
        value = self.input(value)
        for block in self.blocks:
            value = block(value, condition)
        delta = 4.5 * torch.tanh(self.output(F.gelu(value)).squeeze(1))
        return F.softmax((base_log + delta).flatten(1), dim=1).reshape_as(base)


def to_tensor(value: np.ndarray, ids: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value[ids], dtype=np.float32)).to(device)


def batch_inputs(
    ids: np.ndarray,
    data: dict[str, np.ndarray],
    point: np.ndarray,
    v443: np.ndarray,
    v445: np.ndarray,
    device: torch.device,
    *,
    path_ablation: bool = False,
) -> tuple[torch.Tensor, ...]:
    raw = to_tensor(data["train_raw"], ids, device).reshape(-1, 192, 16)
    base = to_tensor(data["train_base"], ids, device).reshape(-1, 192, 16)
    quality = to_tensor(data["train_cell_quality"], ids, device).reshape(-1, 192, 16)
    v443_tensor = to_tensor(v443, ids, device)
    v445_tensor = v443_tensor if path_ablation else to_tensor(v445, ids, device)
    point_tensor = to_tensor(point, ids, device)
    target = to_tensor(data["train_target"], ids, device).reshape(-1, 192, 16)
    return raw, base, quality, v443_tensor, v445_tensor, point_tensor, target


@torch.no_grad()
def predict_train(
    model: MultiViewAdapter,
    ids: np.ndarray,
    data: dict[str, np.ndarray],
    point: np.ndarray,
    v443: np.ndarray,
    v445: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    path_ablation: bool = False,
) -> tuple[np.ndarray, float]:
    model.eval()
    output = []
    cosines = []
    for start in range(0, len(ids), batch_size):
        row = ids[start : start + batch_size]
        raw, base, quality, view443, view445, point_tensor, target = batch_inputs(
            row, data, point, v443, v445, device, path_ablation=path_ablation
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(raw, base, quality, view443, view445, point_tensor)
        output.append(prediction.float().cpu().numpy())
        cosines.extend(
            F.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=1)
            .float()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        )
    return np.concatenate(output).astype(np.float32), float(np.mean(cosines))


@torch.no_grad()
def predict_test(
    model: MultiViewAdapter,
    data: dict[str, np.ndarray],
    point: np.ndarray,
    v443: np.ndarray,
    v445: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    rows = []
    for start in range(0, 500, batch_size):
        end = min(start + batch_size, 500)
        ids = np.arange(start, end, dtype=np.int64)
        raw = torch.from_numpy(data["test_raw"][ids].astype(np.float32)).to(device).reshape(-1, 192, 16)
        base = torch.from_numpy(data["test_base"][ids].astype(np.float32)).to(device).reshape(-1, 192, 16)
        quality = torch.from_numpy(data["test_cell_quality"][ids].astype(np.float32)).to(device).reshape(-1, 192, 16)
        view443 = torch.from_numpy(np.asarray(v443[2000 + ids], dtype=np.float32)).to(device)
        view445 = torch.from_numpy(np.asarray(v445[2000 + ids], dtype=np.float32)).to(device)
        point_tensor = torch.from_numpy(point[2000 + ids]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(raw, base, quality, view443, view445, point_tensor)
        rows.append(prediction.float().cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.reshape(len(left), -1).astype(np.float64)
    right = right.reshape(len(right), -1).astype(np.float64)
    return np.sum(left * right, axis=1) / np.maximum(
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12
    )


def train_fold(
    fold: int,
    fit_ids: np.ndarray,
    validation_ids: np.ndarray,
    data: dict[str, np.ndarray],
    point: np.ndarray,
    v443: np.ndarray,
    v445: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[MultiViewAdapter, dict[str, object]]:
    torch.manual_seed(args.seed + fold)
    rng = np.random.default_rng(args.seed + fold)
    model = MultiViewAdapter(point.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    epochs = 2 if args.smoke else args.epochs
    patience = 2 if args.smoke else args.patience
    best = -math.inf
    best_state = None
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(fit_ids)
        losses = []
        for start in range(0, len(order), args.batch_size):
            row = order[start : start + args.batch_size]
            raw, base, quality, view443, view445, point_tensor, target = batch_inputs(
                row, data, point, v443, v445, device
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model(raw, base, quality, view443, view445, point_tensor)
                loss = v469.spectral_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        _, validation_cosine = predict_train(
            model, validation_ids, data, point, v443, v445, device, args.batch_size
        )
        row_report = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_cosine": validation_cosine,
        }
        history.append(row_report)
        print(f"v478 fold={fold} {json.dumps(row_report)}", flush=True)
        if validation_cosine > best + 1e-6:
            best = validation_cosine
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"Fold {fold} did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint = CHECKPOINT_DIR / f"fold_{fold}.pt"
    torch.save({"model": best_state, "fold": fold, "epoch": best_epoch}, checkpoint)
    normal_prediction, normal_cosine = predict_train(
        model, validation_ids, data, point, v443, v445, device, args.batch_size
    )
    ablated_prediction, ablated_cosine = predict_train(
        model, validation_ids, data, point, v443, v445, device, args.batch_size,
        path_ablation=True,
    )
    return model, {
        "fold": fold,
        "best_epoch": best_epoch,
        "normal_cosine": normal_cosine,
        "path_ablation_cosine": ablated_cosine,
        "path_gain": normal_cosine - ablated_cosine,
        "history": history,
        "checkpoint": PATHS.relative(checkpoint),
        "validation_ids": validation_ids,
        "normal_prediction": normal_prediction,
        "ablated_prediction": ablated_prediction,
    }


def main() -> None:
    args = parse_args()
    for path in (
        v477.V443_JOINT,
        v477.V445_JOINT,
        PATHS.cache_dir(473) / "v473_train_neural_joint_target.npy",
        v469.JOINT_PRED,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    data = v469.build_joint_dataset()
    point = np.load(v469.POINT_FEATURE_CACHE).astype(np.float32)
    view443 = np.load(v477.V443_JOINT, mmap_mode="r")
    view445 = np.load(v477.V445_JOINT, mmap_mode="r")
    fold_id = data["train_fold"].astype(np.int64)
    folds = sorted(np.unique(fold_id).tolist())
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"Unexpected folds {folds}")
    device = torch.device(args.device)
    oof = np.zeros((2000, 192, 16), dtype=np.float32)
    oof_ablation = np.zeros_like(oof)
    test_predictions = []
    fold_reports = []
    for fold in folds:
        validation_ids = np.flatnonzero(fold_id == fold).astype(np.int64)
        fit_ids = np.flatnonzero(fold_id != fold).astype(np.int64)
        model, result = train_fold(
            fold, fit_ids, validation_ids, data, point, view443, view445,
            device, args
        )
        oof[validation_ids] = result.pop("normal_prediction")
        oof_ablation[validation_ids] = result.pop("ablated_prediction")
        result.pop("validation_ids")
        test_predictions.append(
            predict_test(model, data, point, view443, view445, device, args.batch_size)
        )
        fold_reports.append(result)
        del model
        torch.cuda.empty_cache()
    np.save(OOF, oof)
    np.save(OOF_ABLATION, oof_ablation)
    test_target = np.mean(np.stack(test_predictions, axis=0), axis=0).astype(np.float32)
    np.save(TEST_TARGET, test_target)

    truth = data["train_target"].reshape(2000, 192, 16).astype(np.float32)
    base = data["train_base"].reshape(2000, 192, 16).astype(np.float32)
    oof_cosine = float(np.mean(cosine_rows(oof, truth)))
    ablation_cosine = float(np.mean(cosine_rows(oof_ablation, truth)))
    base_cosine = float(np.mean(cosine_rows(base, truth)))
    train_anchor = np.load(
        PATHS.cache_dir(473) / "v473_train_neural_joint_target.npy", mmap_mode="r"
    ).reshape(2000, 192, 16)
    blend_rows = []
    best_alpha = 0.0
    best_score = -math.inf
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = v477.geometric_blend(train_anchor, oof, float(alpha))
        score = float(np.mean(cosine_rows(blended, truth)))
        blend_rows.append({"alpha": float(alpha), "joint_sh_cosine": score})
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    test_anchor = np.load(v469.JOINT_PRED, mmap_mode="r").reshape(500, 192, 16)
    blended_test = v477.geometric_blend(test_anchor, test_target, best_alpha)
    np.save(BLENDED_TARGET, blended_test)
    old_output = v477.OUTPUT
    v477.OUTPUT = OUTPUT
    try:
        v477.generate_channel(blended_test)
    finally:
        v477.OUTPUT = old_output
    replay = v208.score_prediction(OUTPUT)
    baseline_replay = v208.score_prediction(v469.OUTPUT)
    decision = (
        oof_cosine - base_cosine >= 0.02
        and oof_cosine - ablation_cosine >= 0.001
        and replay["score_traceback_leaked"] > baseline_replay["score_traceback_leaked"]
    )
    report = {
        "method": VERSION,
        "status": "candidate_signal_passed" if decision else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Cross-fit official adaptation can retain only the transferable part of the "
            "v445 same-scene path-composition view while anchoring to v469 spectra."
        ),
        "parameters": vars(args),
        "folds": fold_reports,
        "oof": {
            "base_cosine": base_cosine,
            "adapter_cosine": oof_cosine,
            "path_ablation_cosine": ablation_cosine,
            "adapter_minus_base": oof_cosine - base_cosine,
            "adapter_minus_path_ablation": oof_cosine - ablation_cosine,
        },
        "blend_search": blend_rows,
        "selected_alpha": best_alpha,
        "selected_train_cosine": best_score,
        "replay": replay,
        "baseline_v469_replay": baseline_replay,
        "replay_delta_score": float(
            replay["score_traceback_leaked"] - baseline_replay["score_traceback_leaked"]
        ),
        "decision": decision,
        "outputs": {
            "oof": PATHS.relative(OOF),
            "oof_ablation": PATHS.relative(OOF_ABLATION),
            "test_target": PATHS.relative(TEST_TARGET),
            "blended_target": PATHS.relative(BLENDED_TARGET),
            "channel": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
