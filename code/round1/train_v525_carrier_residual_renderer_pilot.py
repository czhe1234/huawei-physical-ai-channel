#!/usr/bin/env python3
"""Train path gains against the PDP residual left after the v447/v458 calibration."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

import analyze_v447_sionna_linkwise_pdp_shape_calibration as v447
import analyze_v501_full_official_sionna_inverse_calibration as v501
import generate_v524_carrier_aware_path_pdp_residual as v524
import train_v519_end_to_end_path_spectral_renderer as v519
from project_paths import PATHS


VERSION_NUMBER = 525
VERSION = "v525_carrier_residual_renderer_pilot"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
FOLD_DIR = CACHE_DIR / "fold_results"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
COEFFICIENTS = PATHS.cache_dir(520) / "v520_single_config_coefficients_points160.npz"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
REGIME_NAMES = v524.REGIME_NAMES
TINY = 1e-20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--max-log-gain", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=525_001)
    return parser.parse_args()


def calibrated_target_for_fold(
    point_ids: np.ndarray,
    fold: int,
    all_fold_codes: np.ndarray,
    all_regimes: np.ndarray,
) -> np.ndarray:
    truth = np.load(v447.TRUTH_CACHE, mmap_mode="r")
    source = np.load(v447.SOURCE_CACHE, mmap_mode="r")
    fit_ids = np.flatnonzero(all_fold_codes != fold).astype(np.int64)
    regime_names = np.asarray(REGIME_NAMES)[all_regimes[:2000]]
    output = np.empty((len(point_ids), 1024, 192), dtype=np.float16)
    for start in range(0, source.shape[1], 16):
        end = min(start + 16, source.shape[1])
        ratios = v447.fit_ratio_chunk(
            truth, source, fit_ids, regime_names, start, end
        )
        values = np.asarray(source[point_ids, start:end], dtype=np.float32)
        local = np.empty_like(values)
        for code, name in enumerate(REGIME_NAMES):
            mask = all_regimes[point_ids] == code
            if np.any(mask):
                local[mask] = v524.normalize_links(
                    values[mask] * np.exp(ratios[name])[None]
                )
        output[:, start:end] = local.astype(np.float16)
    return output.reshape(len(point_ids), 256, 4, 192)


def render_from_gains(
    tau: torch.Tensor,
    h_center: torch.Tensor,
    source: torch.Tensor,
    gains: torch.Tensor,
) -> torch.Tensor:
    corrected = source.reshape(-1, 32, 2, 8, 4)
    corrected = corrected * torch.exp(gains.reshape(-1, 32, 2, 8, 1))
    frequency = (
        torch.arange(192, device=tau.device, dtype=torch.float32) - 96
    ) * 30_000.0
    delay = torch.exp(-2j * math.pi * frequency[None, :, None] * tau[:, None, :])
    element = torch.arange(16, device=tau.device, dtype=torch.float32)
    horizontal = torch.exp(
        2j * math.pi * element[None, :, None] * h_center[:, None, :] / 16.0
    )
    basis = delay[:, :, None, :] * horizontal[:, None, :, :]
    return torch.einsum("bfhk,bkpvu->bpfhvu", basis, corrected)


def normalized_link_pdp(prediction: torch.Tensor) -> torch.Tensor:
    channel = prediction.permute(0, 1, 3, 4, 5, 2)
    power = torch.abs(torch.fft.ifft(channel, dim=-1)) ** 2
    power = power / torch.clamp(torch.sum(power, dim=-1, keepdim=True), min=TINY)
    return power.reshape(len(power), 256, 4, 192)


def corrected_carrier_pdp(
    old: torch.Tensor,
    source_pdp: torch.Tensor,
    corrected_pdp: torch.Tensor,
) -> torch.Tensor:
    floor = 1e-6 / 192.0
    ratio = torch.clamp(
        torch.log(torch.clamp(corrected_pdp, min=floor))
        - torch.log(torch.clamp(source_pdp, min=floor)),
        -2.0,
        2.0,
    )
    candidate = old * torch.exp(ratio)
    return candidate / torch.clamp(torch.sum(candidate, dim=-1, keepdim=True), min=TINY)


def point_score(candidate: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = torch.sum(candidate * truth, dim=-1)
    left = torch.sum(candidate * candidate, dim=-1)
    right = torch.sum(truth * truth, dim=-1)
    cosine = numerator / torch.sqrt(torch.clamp(left * right, min=TINY))
    return torch.mean(cosine, dim=(1, 2))


def batch_tensors(
    ids: np.ndarray,
    arrays: dict[str, np.ndarray],
    features: np.ndarray,
    old: np.ndarray,
    truth: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(features[ids]).to(device),
        torch.from_numpy(arrays["valid"][ids]).to(device),
        torch.from_numpy(arrays["tau"][ids].astype(np.float32)).to(device),
        torch.from_numpy(arrays["h_center"][ids].astype(np.float32)).to(device),
        torch.from_numpy(arrays["source_coefficients"][ids]).to(device),
        torch.from_numpy(np.asarray(old[ids], dtype=np.float32)).to(device),
        torch.from_numpy(np.asarray(truth[ids], dtype=np.float32)).to(device),
    )


@torch.no_grad()
def evaluate(
    model: v519.PathGainTransformer,
    ids: np.ndarray,
    arrays: dict[str, np.ndarray],
    features: np.ndarray,
    old: np.ndarray,
    truth: np.ndarray,
    device: torch.device,
    batch_size: int,
    shuffled_ids: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(ids), batch_size):
        local = ids[start : start + batch_size]
        tensors = batch_tensors(local, arrays, features, old, truth, device)
        gain_features = tensors[0]
        gain_valid = tensors[1]
        if shuffled_ids is not None:
            geometry = shuffled_ids[start : start + len(local)]
            gain_features = torch.from_numpy(features[geometry]).to(device)
            gain_valid = torch.from_numpy(arrays["valid"][geometry]).to(device)
        gains = model.gains(gain_features, gain_valid)
        zero = torch.zeros_like(gains)
        source = normalized_link_pdp(
            render_from_gains(tensors[2], tensors[3], tensors[4], zero)
        )
        corrected = normalized_link_pdp(
            render_from_gains(tensors[2], tensors[3], tensors[4], gains)
        )
        candidate = corrected_carrier_pdp(tensors[5], source, corrected)
        output.extend(point_score(candidate, tensors[6]).cpu().numpy().tolist())
    return np.asarray(output, dtype=np.float32)


def main() -> None:
    args = parse_args()
    for path in (COEFFICIENTS, TRAIN_POS, v447.TRUTH_CACHE, v447.SOURCE_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(COEFFICIENTS) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    point_ids = arrays["indices"].astype(np.int64)
    positions = np.load(TRAIN_POS).astype(np.float32)
    all_regimes = v524.regimes_all()
    all_fold_codes = v501.spatial_fold_codes(positions, all_regimes[:2000])
    fold_codes = all_fold_codes[point_ids]
    regimes = all_regimes[point_ids]
    raw_features = v524.v520_features(arrays, point_ids, regimes)
    truth_all = np.load(v447.TRUTH_CACHE, mmap_mode="r")
    truth = np.asarray(truth_all[point_ids], dtype=np.float32).reshape(
        len(point_ids), 256, 4, 192
    )
    device = torch.device(args.device)
    oof = np.zeros(len(point_ids), dtype=np.float32)
    shuffled = np.zeros(len(point_ids), dtype=np.float32)
    baseline = np.zeros(len(point_ids), dtype=np.float32)
    fold_rows = []
    FOLD_DIR.mkdir(parents=True, exist_ok=True)

    for fold in sorted(np.unique(fold_codes).tolist()):
        fit_ids = np.flatnonzero(fold_codes != fold).astype(np.int64)
        validation_ids = np.flatnonzero(fold_codes == fold).astype(np.int64)
        fold_cache = FOLD_DIR / f"fold_{int(fold)}.npz"
        if fold_cache.is_file():
            with np.load(fold_cache) as loaded:
                saved_ids = loaded["validation_ids"].astype(np.int64)
                if not np.array_equal(saved_ids, validation_ids):
                    raise ValueError(f"fold cache mismatch: {fold_cache}")
                baseline[validation_ids] = loaded["baseline"]
                oof[validation_ids] = loaded["normal"]
                shuffled[validation_ids] = loaded["shuffled"]
                fold_rows.append(json.loads(str(loaded["details"].item())))
            print(f"reuse v525 fold {fold}", flush=True)
            continue
        old = calibrated_target_for_fold(
            point_ids, int(fold), all_fold_codes, all_regimes
        ).astype(np.float32)
        baseline[validation_ids] = np.mean(
            v524.link_cosine(
                truth[validation_ids].reshape(len(validation_ids), 1024, 192),
                old[validation_ids].reshape(len(validation_ids), 1024, 192),
            ),
            axis=1,
        )
        selected = raw_features[fit_ids][arrays["valid"][fit_ids]]
        mean = selected.mean(axis=0, keepdims=True).astype(np.float32)
        std = np.maximum(selected.std(axis=0, keepdims=True), 1e-4).astype(np.float32)
        features = ((raw_features - mean) / std).astype(np.float32)
        features *= arrays["valid"][..., None]
        torch.manual_seed(args.seed + int(fold))
        rng = np.random.default_rng(args.seed + int(fold))
        model = v519.PathGainTransformer(features.shape[2], args.max_log_gain).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        best_score = float(np.mean(baseline[validation_ids]))
        best_epoch = 0
        best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        stale = 0
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = rng.permutation(fit_ids)
            running = 0.0
            for start in range(0, len(order), args.batch_size):
                ids = order[start : start + args.batch_size]
                tensors = batch_tensors(ids, arrays, features, old, truth, device)
                gains = model.gains(tensors[0], tensors[1])
                zero = torch.zeros_like(gains)
                source = normalized_link_pdp(
                    render_from_gains(tensors[2], tensors[3], tensors[4], zero)
                )
                corrected = normalized_link_pdp(
                    render_from_gains(tensors[2], tensors[3], tensors[4], gains)
                )
                candidate = corrected_carrier_pdp(tensors[5], source, corrected)
                score = point_score(candidate, tensors[6])
                loss = 1.0 - torch.mean(score) + 1e-4 * torch.mean(gains * gains)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running += float(loss.detach().cpu()) * len(ids)
            if epoch == 1 or epoch % 5 == 0:
                values = evaluate(
                    model, validation_ids, arrays, features, old, truth,
                    device, args.batch_size
                )
                mean_score = float(np.mean(values))
                history.append(
                    {"epoch": epoch, "train_loss": running / len(fit_ids), "validation": mean_score}
                )
                if mean_score > best_score + 1e-5:
                    best_score = mean_score
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
        model.load_state_dict(best_state)
        oof[validation_ids] = evaluate(
            model, validation_ids, arrays, features, old, truth,
            device, args.batch_size
        )
        shuffled_ids = validation_ids[
            np.random.default_rng(args.seed + 10_000 + int(fold)).permutation(len(validation_ids))
        ]
        shuffled[validation_ids] = evaluate(
            model, validation_ids, arrays, features, old, truth,
            device, args.batch_size, shuffled_ids=shuffled_ids
        )
        fold_detail = {
                "fold": int(fold),
                "count": int(len(validation_ids)),
                "best_epoch": int(best_epoch),
                "baseline": float(np.mean(baseline[validation_ids])),
                "normal": float(np.mean(oof[validation_ids])),
                "shuffled": float(np.mean(shuffled[validation_ids])),
                "gain": float(np.mean(oof[validation_ids] - baseline[validation_ids])),
                "margin_vs_shuffled": float(np.mean(oof[validation_ids] - shuffled[validation_ids])),
                "history": history,
            }
        fold_rows.append(fold_detail)
        np.savez_compressed(
            fold_cache,
            validation_ids=validation_ids,
            baseline=baseline[validation_ids],
            normal=oof[validation_ids],
            shuffled=shuffled[validation_ids],
            details=np.asarray(json.dumps(fold_detail)),
        )
        print(json.dumps(fold_detail), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gain = oof - baseline
    margin = oof - shuffled
    passed = bool(
        np.mean(gain) >= 0.005
        and np.sum([row["gain"] > 0.0 for row in fold_rows]) >= 4
        and np.mean(margin) >= 0.003
        and np.sum([row["margin_vs_shuffled"] > 0.0 for row in fold_rows]) >= 4
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": "A path renderer trained on the residual after v447 calibration learns information that was duplicated by v523's raw-source objective.",
        "point_count": int(len(point_ids)),
        "model": {
            "type": "four-layer path-set Transformer with carrier-aware differentiable PDP residual renderer",
            "epochs": args.epochs,
            "patience": args.patience,
            "max_log_gain": args.max_log_gain,
        },
        "oof": {
            "baseline": float(np.mean(baseline)),
            "normal": float(np.mean(oof)),
            "shuffled": float(np.mean(shuffled)),
            "gain": float(np.mean(gain)),
            "margin_vs_shuffled": float(np.mean(margin)),
            "positive_folds": int(sum(row["gain"] > 0.0 for row in fold_rows)),
            "positive_margin_folds": int(sum(row["margin_vs_shuffled"] > 0.0 for row in fold_rows)),
            "by_regime": {
                name: {
                    "count": int(np.sum(regimes == code)),
                    "gain": float(np.mean(gain[regimes == code])),
                    "margin_vs_shuffled": float(np.mean(margin[regimes == code])),
                }
                for code, name in enumerate(REGIME_NAMES)
            },
        },
        "folds": fold_rows,
        "deployment_gate": {
            "pass": passed,
            "minimum_gain": 0.005,
            "minimum_positive_folds": 4,
            "minimum_margin": 0.003,
            "minimum_positive_margin_folds": 4,
        },
        "outputs": {"report": PATHS.relative(REPORT)},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
