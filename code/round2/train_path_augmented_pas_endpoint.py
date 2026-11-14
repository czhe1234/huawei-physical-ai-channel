#!/usr/bin/env python3
"""Train a V9-compatible PAS endpoint augmented by explicit-path context."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
import time

import numpy as np
from sklearn.decomposition import PCA
import torch

from project_paths import PATHS
from train_sionna_spectral_calibrator import (
    SpectralCalibrator,
    TrainConfig,
    evaluate_ids,
    fold_ids,
    reshape_oof_prediction,
    train_epochs,
)
from train_v9_joint_sionna_endpoints import (
    JointDomainBank,
    MODE,
    OOF_META,
    instantiate_model,
    predict_model,
)


VERSION = "Round2-E71-path-augmented-PAS-endpoint"
FEATURE_DIR = PATHS.artifacts / "round2" / "explicit_path_pdp_predictor"
TRAIN_PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n4000_s24_k128_b096_d064_scatter1_sel106"
)
TEST_PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "test_n500_s24_k128_b096_d064_scatter1_sel106"
)
ARTIFACT_DIR = PATHS.artifacts / "round2" / "path_augmented_pas_endpoint"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_path_augmented_pas_endpoint"
TRAIN_CONTEXT = ARTIFACT_DIR / "train_path_context.npy"
TEST_CONTEXT = ARTIFACT_DIR / "test_path_context.npy"
PCA_STATE = ARTIFACT_DIR / "path_context_pca.npz"
OOF_OUTPUT = ARTIFACT_DIR / "oof_pas_path_augmented_f16.npy"
TEST_OUTPUT = ARTIFACT_DIR / "test_pas_path_augmented_f16.npy"
OOF_REPORT = ANALYSIS_DIR / "path_augmented_pas_oof.json"
TEST_REPORT = ANALYSIS_DIR / "path_augmented_pas_test.json"
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("context", "oof", "test", "all"), default="all"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--steps-per-epoch", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-batch-size", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--context-channels", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--residual-scale", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=8e-5)
    parser.add_argument("--hellinger-weight", type=float, default=0.08)
    parser.add_argument("--residual-weight", type=float, default=3e-4)
    parser.add_argument("--pca-components", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026080771)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_folds(value: str) -> tuple[int, ...]:
    folds = tuple(int(item) for item in value.split(",") if item.strip())
    if not folds or any(fold < 0 or fold > 4 for fold in folds):
        raise ValueError("folds must be a subset of 0,1,2,3,4")
    return folds


def raw_path_summary(split: str) -> np.ndarray:
    if split == "train":
        features_path = FEATURE_DIR / "path_features_train_n4000.npy"
        cache = TRAIN_PATH_CACHE
    elif split == "test":
        features_path = FEATURE_DIR / "path_features_test_n500.npy"
        cache = TEST_PATH_CACHE
    else:
        raise ValueError(split)
    features = np.load(features_path, mmap_mode="r")
    mask = np.asarray(np.load(cache / "mask.npy", mmap_mode="r"), dtype=bool)
    fraction = np.asarray(
        np.load(cache / "energy_fraction.npy", mmap_mode="r"), dtype=np.float32
    )
    output = np.empty((len(features), features.shape[2] * 3), dtype=np.float32)
    for start in range(0, len(features), 32):
        end = min(start + 32, len(features))
        local = np.asarray(features[start:end], dtype=np.float32)
        valid = mask[start:end]
        valid_float = valid[..., None].astype(np.float32)
        mean = (local * valid_float).sum(axis=1) / valid_float.sum(axis=1).clip(min=1.0)
        weight = np.where(valid, fraction[start:end], 0.0)
        weight /= weight.sum(axis=1, keepdims=True).clip(min=1e-12)
        weighted = np.einsum("bk,bkf->bf", weight, local, optimize=True)
        top = np.argpartition(-weight, kth=min(7, weight.shape[1] - 1), axis=1)[:, :8]
        top_values = np.take_along_axis(local, top[..., None], axis=1)
        top_valid = np.take_along_axis(valid, top, axis=1)[..., None].astype(np.float32)
        top_mean = (top_values * top_valid).sum(axis=1) / top_valid.sum(axis=1).clip(min=1.0)
        output[start:end] = np.concatenate((mean, weighted, top_mean), axis=1)
    return output


def build_path_context(components: int, force: bool) -> None:
    if TRAIN_CONTEXT.is_file() and TEST_CONTEXT.is_file() and not force:
        return
    train = raw_path_summary("train")
    test = raw_path_summary("test")
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    train_scaled = (train - mean) / std
    test_scaled = (test - mean) / std
    model = PCA(n_components=components, whiten=True, random_state=2026080771)
    train_context = model.fit_transform(train_scaled).astype(np.float32)
    test_context = model.transform(test_scaled).astype(np.float32)
    np.save(TRAIN_CONTEXT, train_context)
    np.save(TEST_CONTEXT, test_context)
    np.savez_compressed(
        PCA_STATE,
        mean=mean,
        std=std,
        components=model.components_.astype(np.float32),
        explained_variance=model.explained_variance_.astype(np.float32),
        explained_variance_ratio=model.explained_variance_ratio_.astype(np.float32),
    )
    print(
        f"path context train={train_context.shape} test={test_context.shape} "
        f"explained={model.explained_variance_ratio_.sum():.6f}",
        flush=True,
    )


class PathAugmentedPasBank(JointDomainBank):
    def __init__(self, split: str, device: torch.device, force: bool = False) -> None:
        super().__init__("pas", split, device, force)
        context_path = TRAIN_CONTEXT if split == "train" else TEST_CONTEXT
        path_context = torch.as_tensor(
            np.load(context_path, mmap_mode="r"), device=device, dtype=torch.float32
        )
        self._point_context = torch.cat((self._point_context, path_context), dim=1)
        print(
            f"augmented PAS context split={split} dim={self._point_context.shape[1]}",
            flush=True,
        )


def train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        pas_batch_size=args.batch_size,
        pdp_batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        hidden=args.hidden,
        context_channels=args.context_channels,
        blocks=args.blocks,
        residual_scale=args.residual_scale,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hellinger_weight=args.hellinger_weight,
        residual_weight=args.residual_weight,
        seed=args.seed,
    )


def checkpoint_path(fold: int) -> object:
    return ARTIFACT_DIR / f"path_augmented_pas_fold{fold}.pt"


def save_checkpoint(
    fold: int,
    epoch: int,
    model: SpectralCalibrator,
    config: TrainConfig,
) -> None:
    torch.save(
        {
            "version": VERSION,
            "domain": "pas",
            "mode": MODE,
            "fold": fold,
            "epoch": epoch,
            "config": asdict(config),
            "input_channels": int(model.stem.in_channels),
            "state_dict": model.state_dict(),
            "context_mean": model.context_mean.detach().cpu(),
            "context_std": model.context_std.detach().cpu(),
        },
        checkpoint_path(fold),
    )


def run_oof(
    folds: tuple[int, ...],
    device: torch.device,
    config: TrainConfig,
    force: bool,
) -> dict[str, object]:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    row_for_id = {int(point_id): row for row, point_id in enumerate(query_ids)}
    bank = PathAugmentedPasBank("train", device, force)
    output = np.lib.format.open_memmap(
        OOF_OUTPUT, mode="w+", dtype=np.float16, shape=(500, 2, 16, 8, 4, 192)
    )
    output[...] = 0.0
    nonzero = np.asarray(bank.nonzero.cpu())
    rows = []
    for fold in folds:
        outer_fit, outer_query, nested_fit, validation = fold_ids(fold)
        screening_model, best_epoch, history = train_epochs(
            bank,
            MODE,
            nested_fit,
            validation,
            config.epochs,
            config,
            config.seed + 1000 * fold,
        )
        validation_score, _ = evaluate_ids(
            screening_model,
            bank,
            validation[nonzero[validation]],
            MODE,
            screening_model.context_mean,
            screening_model.context_std,
            config.validation_batch_size,
            False,
        )
        del screening_model
        model, _, _ = train_epochs(
            bank,
            MODE,
            outer_fit,
            None,
            best_epoch,
            config,
            config.seed + 10_000 + 1000 * fold,
        )
        held_nonzero = outer_query[nonzero[outer_query]]
        held_score, prediction = evaluate_ids(
            model,
            bank,
            held_nonzero,
            MODE,
            model.context_mean,
            model.context_std,
            config.validation_batch_size,
            True,
        )
        output_rows = np.asarray([row_for_id[int(value)] for value in held_nonzero])
        output[output_rows] = reshape_oof_prediction("pas", prediction)
        output.flush()
        save_checkpoint(fold, best_epoch, model, config)
        rows.append(
            {
                "fold": fold,
                "fit_count": int(len(outer_fit)),
                "nested_fit_count": int(len(nested_fit)),
                "validation_count": int(len(validation)),
                "held_count": int(len(outer_query)),
                "held_nonzero_count": int(len(held_nonzero)),
                "best_epoch": int(best_epoch),
                "validation_cosine": float(validation_score),
                "held_cosine": float(held_score),
                "checkpoint": PATHS.relative(checkpoint_path(fold)),
                "history": history,
            }
        )
        del model, prediction
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output.flush()
    return {
        "folds": rows,
        "mean_held_cosine": float(
            np.average(
                [row["held_cosine"] for row in rows],
                weights=[row["held_nonzero_count"] for row in rows],
            )
        ),
        "output": PATHS.relative(OOF_OUTPUT),
    }


def generate_test(device: torch.device, batch_size: int, force: bool) -> dict[str, object]:
    bank = PathAugmentedPasBank("test", device, force)
    total = None
    checkpoints = []
    for fold in range(5):
        path = checkpoint_path(fold)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model, context_mean, context_std = instantiate_model(checkpoint, bank, device)
        prediction = predict_model(
            model, bank, context_mean, context_std, batch_size
        )
        total = prediction if total is None else total + prediction
        checkpoints.append(
            {"fold": fold, "path": PATHS.relative(path), "epoch": checkpoint["epoch"]}
        )
        del model, prediction
        if device.type == "cuda":
            torch.cuda.empty_cache()
    total /= 5.0
    total /= np.maximum(total.sum(axis=(2, 3, 4), keepdims=True), TINY)
    shaped = reshape_oof_prediction("pas", total)
    output = np.lib.format.open_memmap(
        TEST_OUTPUT, mode="w+", dtype=np.float16, shape=shaped.shape
    )
    output[...] = shaped.astype(np.float16)
    output.flush()
    return {
        "output": PATHS.relative(TEST_OUTPUT),
        "checkpoints": checkpoints,
    }


def write_report(path: object, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (
        FEATURE_DIR / "path_features_train_n4000.npy",
        FEATURE_DIR / "path_features_test_n500.npy",
        TRAIN_PATH_CACHE,
        TEST_PATH_CACHE,
        OOF_META,
    )
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    started = time.time()
    build_path_context(args.pca_components, args.force)
    if args.phase == "context":
        return
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    config = train_config(args)
    common = {
        "experiment": VERSION,
        "primary_hypothesis": (
            "Explicit paths are more useful as context for an official-H-supervised PAS "
            "endpoint than as a raw PAS target."
        ),
        "configuration": vars(args),
        "train_context": PATHS.relative(TRAIN_CONTEXT),
        "test_context": PATHS.relative(TEST_CONTEXT),
    }
    if args.phase in ("oof", "all"):
        result = run_oof(parse_folds(args.folds), device, config, args.force)
        write_report(
            OOF_REPORT,
            {
                **common,
                "phase": "oof",
                "status": "complete",
                "result": result,
                "seconds": float(time.time() - started),
            },
        )
        if args.phase == "oof":
            return
    if args.phase in ("test", "all"):
        result = generate_test(device, args.validation_batch_size, args.force)
        write_report(
            TEST_REPORT,
            {
                **common,
                "phase": "test",
                "status": "complete",
                "result": result,
                "seconds": float(time.time() - started),
            },
        )


if __name__ == "__main__":
    main()
