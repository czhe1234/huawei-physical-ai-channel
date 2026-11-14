#!/usr/bin/env python3
"""Train path corrections end-to-end against final PAS/PDP instead of unstable coefficients."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.fft import fft2, ifft
import torch
import torch.nn as nn

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v516_shared_complex_path_correction as v516
import train_v518_path_pol_vertical_correction as v518
from project_paths import PATHS


VERSION_NUMBER = 519
VERSION = "v519_end_to_end_path_spectral_renderer"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
FOLD_DIR = CACHE_DIR / "fold_predictions"
REPORT = REPORT_DIR / f"{VERSION}.json"
TARGET_CACHE = CACHE_DIR / "v519_official_target_spectra_points160.npz"
POINT_METRICS = CACHE_DIR / "v519_point_metrics.npz"
COEFFICIENT_CACHE = PATHS.cache_dir(516) / "v516_path_coefficients_points160.npz"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"

NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--max-log-gain", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=519_001)
    parser.add_argument("--fold", type=int, choices=range(5))
    return parser.parse_args()


def normalize_l2(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    norm = np.sqrt(np.sum(value * value, axis=axes, keepdims=True, dtype=np.float64))
    return (value / np.maximum(norm, TINY)).astype(np.float32)


def build_target_cache(point_ids: np.ndarray) -> dict[str, np.ndarray]:
    if TARGET_CACHE.is_file():
        with np.load(TARGET_CACHE) as loaded:
            return {name: np.asarray(loaded[name]) for name in loaded.files}
    truth = np.load(TRAIN_H, mmap_mode="r")
    pdp = np.empty((len(point_ids), 2, 16, 8, 4, 192), dtype=np.float16)
    pas = np.empty((len(point_ids), 2, 192, 16, 4, 8), dtype=np.float16)
    for row, point_id in enumerate(point_ids):
        channel = np.asarray(truth[int(point_id)], dtype=np.complex64).reshape(
            2, 16, 8, 4, 192
        )
        delay_power = np.abs(ifft(channel, axis=-1, workers=1)).astype(np.float32) ** 2
        pdp[row] = normalize_l2(delay_power, (4,)).astype(np.float16)
        leaked = channel.transpose(0, 4, 1, 3, 2)
        leaked_power = np.abs(fft2(leaked, axes=(1, 2), workers=1)).astype(np.float32) ** 2
        pas[row] = normalize_l2(leaked_power, (1, 2)).astype(np.float16)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TARGET_CACHE, indices=point_ids, pdp=pdp, pas=pas)
    return {"indices": point_ids, "pdp": pdp, "pas": pas}


class PathGainTransformer(nn.Module):
    def __init__(self, feature_dim: int, max_log_gain: float) -> None:
        super().__init__()
        width = 160
        self.max_log_gain = max_log_gain
        self.input = nn.Sequential(
            nn.Linear(feature_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=8,
            dim_feedforward=4 * width,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.output = nn.Linear(width, 16)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def gains(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        token = self.input(features)
        safe_valid = valid.clone()
        empty = ~torch.any(safe_valid, dim=1)
        if torch.any(empty):
            safe_valid[empty, 0] = True
        token = self.transformer(token, src_key_padding_mask=~safe_valid)
        value = self.max_log_gain * torch.tanh(self.output(token))
        return value * valid[..., None]

    def forward(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        tau: torch.Tensor,
        h_center: torch.Tensor,
        source_coefficients: torch.Tensor,
        *,
        gain_features: torch.Tensor | None = None,
        gain_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_gain = self.gains(
            features if gain_features is None else gain_features,
            valid if gain_valid is None else gain_valid,
        )
        corrected = source_coefficients.reshape(-1, v516.PATH_LIMIT, 2, 8, 4)
        corrected = corrected * torch.exp(log_gain.reshape(-1, v516.PATH_LIMIT, 2, 8, 1))
        frequency = (
            torch.arange(NUM_SUBCARRIERS, device=tau.device, dtype=torch.float32)
            - NUM_SUBCARRIERS // 2
        ) * SUBCARRIER_SPACING
        delay = torch.exp(
            -2j * math.pi * frequency[None, :, None] * tau[:, None, :]
        )
        element = torch.arange(16, device=tau.device, dtype=torch.float32)
        horizontal = torch.exp(
            2j * math.pi * element[None, :, None] * h_center[:, None, :] / 16.0
        )
        basis = delay[:, :, None, :] * horizontal[:, None, :, :]
        # B, polarization, frequency, horizontal, vertical, UE
        return torch.einsum("bfhk,bkpvu->bpfhvu", basis, corrected)


def point_scores(
    prediction: torch.Tensor,
    target_pdp: torch.Tensor,
    target_pas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delay = torch.fft.ifft(prediction.permute(0, 1, 3, 4, 5, 2), dim=-1)
    delay_power = torch.abs(delay) ** 2
    delay_norm = torch.sqrt(
        torch.sum(delay_power * delay_power, dim=-1, keepdim=True) + 1e-20
    )
    delay_unit = delay_power / delay_norm
    pdp = torch.sum(delay_unit * target_pdp, dim=-1).mean(dim=(1, 2, 3, 4))

    leaked = prediction.permute(0, 1, 2, 3, 5, 4)
    leaked_power = torch.abs(torch.fft.fft2(leaked, dim=(2, 3))) ** 2
    leaked_norm = torch.sqrt(
        torch.sum(leaked_power * leaked_power, dim=(2, 3), keepdim=True) + 1e-20
    )
    leaked_unit = leaked_power / leaked_norm
    pas = torch.sum(leaked_unit * target_pas, dim=(2, 3)).mean(dim=(1, 2, 3))
    return pdp, pas, 0.5 * (pdp + pas)


def spectral_loss(
    prediction: torch.Tensor,
    target_pdp: torch.Tensor,
    target_pas: torch.Tensor,
) -> torch.Tensor:
    _, _, score = point_scores(prediction, target_pdp, target_pas)
    return 1.0 - torch.mean(score)


def batch_tensors(
    ids: np.ndarray,
    arrays: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(np.asarray(arrays["features"][ids])).to(device),
        torch.from_numpy(np.asarray(arrays["valid"][ids])).to(device),
        torch.from_numpy(np.asarray(arrays["tau"][ids], dtype=np.float32)).to(device),
        torch.from_numpy(np.asarray(arrays["h_center"][ids], dtype=np.float32)).to(device),
        torch.from_numpy(np.asarray(arrays["source"][ids])).to(device),
        torch.from_numpy(np.asarray(arrays["target_pdp"][ids], dtype=np.float32)).to(device),
        torch.from_numpy(np.asarray(arrays["target_pas"][ids], dtype=np.float32)).to(device),
    )


@torch.no_grad()
def evaluate(
    model: PathGainTransformer,
    ids: np.ndarray,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    *,
    shuffled_features: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    model.eval()
    pdp_values = []
    pas_values = []
    for start in range(0, len(ids), batch_size):
        local_ids = ids[start : start + batch_size]
        features, valid, tau, h_center, source, target_pdp, target_pas = batch_tensors(
            local_ids, arrays, device
        )
        kwargs = {}
        if shuffled_features is not None:
            geometry_ids = shuffled_features[start : start + len(local_ids)]
            kwargs["gain_features"] = torch.from_numpy(
                np.asarray(arrays["features"][geometry_ids])
            ).to(device)
            kwargs["gain_valid"] = torch.from_numpy(
                np.asarray(arrays["valid"][geometry_ids])
            ).to(device)
        prediction = model(
            features, valid, tau, h_center, source, **kwargs
        )
        pdp, pas, _ = point_scores(prediction, target_pdp, target_pas)
        pdp_values.extend(pdp.cpu().numpy().astype(float).tolist())
        pas_values.extend(pas.cpu().numpy().astype(float).tolist())
    pdp_array = np.asarray(pdp_values, dtype=np.float32)
    pas_array = np.asarray(pas_values, dtype=np.float32)
    return {
        "pdp": pdp_array,
        "pas": pas_array,
        "spectral": 0.5 * (pdp_array + pas_array),
        "mean": float(np.mean(0.5 * (pdp_array + pas_array))),
    }


def train_fold(
    fold: int,
    fit_ids: np.ndarray,
    validation_ids: np.ndarray,
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[PathGainTransformer, dict[str, object], np.ndarray, np.ndarray]:
    seed = args.seed + fold
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    train_valid = arrays["valid"][fit_ids]
    selected = arrays["features"][fit_ids][train_valid]
    mean = selected.mean(axis=0, keepdims=True)
    std = np.maximum(selected.std(axis=0, keepdims=True), 1e-4)
    local_arrays = dict(arrays)
    local_arrays["features"] = ((arrays["features"] - mean) / std).astype(np.float32)
    local_arrays["features"] *= arrays["valid"][..., None]

    model = PathGainTransformer(arrays["features"].shape[2], args.max_log_gain).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(seed)
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(fit_ids)
        running = 0.0
        for start in range(0, len(order), args.batch_size):
            ids = order[start : start + args.batch_size]
            tensors = batch_tensors(ids, local_arrays, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*tensors[:5])
            loss = spectral_loss(prediction, tensors[5], tensors[6])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(ids)
        if epoch == 1 or epoch % 5 == 0:
            validation = evaluate(
                model,
                validation_ids,
                local_arrays,
                device,
                args.batch_size,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": running / len(fit_ids),
                    "validation_spectral": validation["mean"],
                }
            )
            if validation["mean"] > best_score + 1e-5:
                best_score = float(validation["mean"])
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 5
                if stale >= args.patience:
                    break
    if best_state is None:
        raise RuntimeError("no model selected")
    model.load_state_dict(best_state)
    return (
        model,
        {
            "best_epoch": best_epoch,
            "best_validation_spectral": best_score,
            "history": history,
        },
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def summarize(value: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(value, dtype=np.float64)[mask]
    return {
        "count": int(len(selected)),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "p10": float(np.quantile(selected, 0.10)),
        "p90": float(np.quantile(selected, 0.90)),
    }


def main() -> None:
    args = parse_args()
    for path in (COEFFICIENT_CACHE, TRAIN_POS, TRAIN_H):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(COEFFICIENT_CACHE) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}
    point_ids = data["indices"].astype(np.int64)
    target = build_target_cache(point_ids)
    all_positions = np.load(TRAIN_POS).astype(np.float32)
    all_regimes = v501.regime_codes()
    regimes = all_regimes[point_ids]
    fold_codes = v501.spatial_fold_codes(all_positions, all_regimes)[point_ids]
    base_features, _, _, _ = v516.path_features(
        data, all_positions[point_ids], regimes
    )
    features = v518.input_features(
        base_features, data["source_coefficients"].astype(np.complex128)
    )
    arrays = {
        "features": features,
        "valid": data["valid"].astype(bool),
        "tau": data["tau"].astype(np.float32),
        "h_center": data["h_center"].astype(np.float32),
        "source": data["source_coefficients"].astype(np.complex64),
        "target_pdp": target["pdp"],
        "target_pas": target["pas"],
    }
    device = torch.device(args.device)
    baseline_model = PathGainTransformer(features.shape[2], args.max_log_gain).to(device)
    all_ids = np.arange(len(point_ids), dtype=np.int64)
    baseline = evaluate(
        baseline_model, all_ids, arrays, device, args.batch_size
    )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    oof_pdp = np.zeros(len(point_ids), dtype=np.float32)
    oof_pas = np.zeros(len(point_ids), dtype=np.float32)
    shuffled_pdp = np.zeros(len(point_ids), dtype=np.float32)
    shuffled_pas = np.zeros(len(point_ids), dtype=np.float32)
    fold_details = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    available_folds = sorted(np.unique(fold_codes).tolist())
    selected_folds = [args.fold] if args.fold is not None else available_folds
    for fold in selected_folds:
        fit_ids = np.flatnonzero(fold_codes != fold).astype(np.int64)
        validation_ids = np.flatnonzero(fold_codes == fold).astype(np.int64)
        fold_path = FOLD_DIR / f"fold_{int(fold)}.npz"
        if fold_path.is_file():
            with np.load(fold_path) as loaded:
                saved_ids = loaded["validation_ids"].astype(np.int64)
                if not np.array_equal(saved_ids, validation_ids):
                    raise ValueError(f"fold cache mismatch: {fold_path}")
                oof_pdp[validation_ids] = loaded["normal_pdp"]
                oof_pas[validation_ids] = loaded["normal_pas"]
                shuffled_pdp[validation_ids] = loaded["shuffled_pdp"]
                shuffled_pas[validation_ids] = loaded["shuffled_pas"]
                fold_details.append(
                    json.loads(str(loaded["details"].item()))
                )
            print(f"reuse v519 fold {fold}", flush=True)
            continue
        model, training, mean, std = train_fold(
            int(fold), fit_ids, validation_ids, arrays, args
        )
        local_arrays = dict(arrays)
        local_arrays["features"] = ((features - mean) / std).astype(np.float32)
        local_arrays["features"] *= arrays["valid"][..., None]
        normal = evaluate(
            model, validation_ids, local_arrays, device, args.batch_size
        )
        rng = np.random.default_rng(args.seed + 10_000 + int(fold))
        shuffled_ids = validation_ids[rng.permutation(len(validation_ids))]
        shuffled = evaluate(
            model,
            validation_ids,
            local_arrays,
            device,
            args.batch_size,
            shuffled_features=shuffled_ids,
        )
        oof_pdp[validation_ids] = normal["pdp"]
        oof_pas[validation_ids] = normal["pas"]
        shuffled_pdp[validation_ids] = shuffled["pdp"]
        shuffled_pas[validation_ids] = shuffled["pas"]
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "feature_mean": mean,
                "feature_std": std,
                "input_dim": features.shape[2],
                "fold": int(fold),
                "max_log_gain": args.max_log_gain,
            },
            CHECKPOINT_DIR / f"fold_{int(fold)}.pt",
        )
        fold_detail = {
                "fold": int(fold),
                "fit_count": int(len(fit_ids)),
                "validation_count": int(len(validation_ids)),
                "baseline_spectral": float(np.mean(baseline["spectral"][validation_ids])),
                "normal_spectral": normal["mean"],
                "shuffled_feature_spectral": shuffled["mean"],
                "gain_vs_baseline": float(
                    normal["mean"] - np.mean(baseline["spectral"][validation_ids])
                ),
                "margin_vs_shuffled": float(normal["mean"] - shuffled["mean"]),
                "training": training,
            }
        fold_details.append(fold_detail)
        np.savez_compressed(
            fold_path,
            validation_ids=validation_ids,
            normal_pdp=normal["pdp"],
            normal_pas=normal["pas"],
            shuffled_pdp=shuffled["pdp"],
            shuffled_pas=shuffled["pas"],
            details=np.asarray(json.dumps(fold_detail)),
        )
        print(
            f"v519 fold {fold}: base={fold_details[-1]['baseline_spectral']:.6f} "
            f"normal={normal['mean']:.6f} shuffled={shuffled['mean']:.6f}",
            flush=True,
        )

    if args.fold is not None:
        print(
            json.dumps(
                {
                    "fold": int(args.fold),
                    "cache": PATHS.relative(FOLD_DIR / f"fold_{int(args.fold)}.npz"),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    if len(fold_details) != len(available_folds):
        raise RuntimeError("all five fold caches are required before finalization")

    oof_spectral = 0.5 * (oof_pdp + oof_pas)
    shuffled_spectral = 0.5 * (shuffled_pdp + shuffled_pas)
    fold_gains = [row["gain_vs_baseline"] for row in fold_details]
    fold_margins = [row["margin_vs_shuffled"] for row in fold_details]
    mean_gain = float(np.mean(oof_spectral - baseline["spectral"]))
    mean_margin = float(np.mean(oof_spectral - shuffled_spectral))
    supported = bool(
        mean_gain >= 0.005
        and np.sum(np.asarray(fold_gains) > 0.0) >= 4
        and mean_margin >= 0.003
        and np.sum(np.asarray(fold_margins) > 0.0) >= 4
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        POINT_METRICS,
        indices=point_ids,
        regimes=regimes,
        fold_codes=fold_codes,
        baseline_pdp=baseline["pdp"],
        baseline_pas=baseline["pas"],
        baseline_spectral=baseline["spectral"],
        oof_pdp=oof_pdp,
        oof_pas=oof_pas,
        oof_spectral=oof_spectral,
        shuffled_pdp=shuffled_pdp,
        shuffled_pas=shuffled_pas,
        shuffled_spectral=shuffled_spectral,
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Per-path polarization-by-vertical amplitude corrections become transferable "
            "when trained end-to-end on final PAS/PDP, avoiding the non-identifiable "
            "coefficient labels rejected by v518."
        ),
        "point_count": int(len(point_ids)),
        "split": "v501 five spatial folds",
        "model": {
            "type": "four-layer path-set Transformer with differentiable complex renderer",
            "input_dim": int(features.shape[2]),
            "output": "32 paths x 2 polarizations x 8 vertical elements real log-gain",
            "phase_correction": False,
            "max_log_gain": args.max_log_gain,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "loss": "equal-weight final linkwise-PDP and leaked full-PAS cosine",
        },
        "baseline": {
            "pdp": float(np.mean(baseline["pdp"])),
            "pas": float(np.mean(baseline["pas"])),
            "spectral": float(np.mean(baseline["spectral"])),
        },
        "oof": {
            "pdp": summarize(oof_pdp, np.ones(len(point_ids), dtype=bool)),
            "pas": summarize(oof_pas, np.ones(len(point_ids), dtype=bool)),
            "spectral": summarize(oof_spectral, np.ones(len(point_ids), dtype=bool)),
            "heavy_spectral": summarize(oof_spectral, regimes == 3),
        },
        "shuffled_feature_control": {
            "spectral": summarize(
                shuffled_spectral, np.ones(len(point_ids), dtype=bool)
            ),
            "heavy_spectral": summarize(shuffled_spectral, regimes == 3),
        },
        "fold_details": fold_details,
        "decision": {
            "oof_gain_vs_source_renderer": mean_gain,
            "positive_folds_vs_source": int(np.sum(np.asarray(fold_gains) > 0.0)),
            "margin_vs_shuffled_features": mean_margin,
            "positive_folds_vs_shuffled": int(
                np.sum(np.asarray(fold_margins) > 0.0)
            ),
            "end_to_end_path_renderer_supported": supported,
            "gate": (
                "gain >= 0.005 and 4/5 positive folds; margin vs shuffled >= 0.003 "
                "and 4/5 positive folds"
            ),
        },
        "outputs": {
            "target_cache": PATHS.relative(TARGET_CACHE),
            "checkpoints": PATHS.relative(CHECKPOINT_DIR),
            "point_metrics": PATHS.relative(POINT_METRICS),
            "report": PATHS.relative(REPORT),
            "submission_zip": None,
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline": report["baseline"],
                "oof": {
                    "pdp": report["oof"]["pdp"]["mean"],
                    "pas": report["oof"]["pas"]["mean"],
                    "spectral": report["oof"]["spectral"]["mean"],
                },
                "gain": mean_gain,
                "positive_folds": report["decision"]["positive_folds_vs_source"],
                "margin_vs_shuffled": mean_margin,
                "supported": supported,
                "report": PATHS.relative(REPORT),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
