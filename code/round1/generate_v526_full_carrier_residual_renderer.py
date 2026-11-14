#!/usr/bin/env python3
"""Train full-fold carrier-aware path residual renderers and deploy their consensus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

import analyze_v447_sionna_linkwise_pdp_shape_calibration as v447
import analyze_v501_full_official_sionna_inverse_calibration as v501
import generate_v523_end_to_end_path_pdp_residual as v523
import generate_v524_carrier_aware_path_pdp_residual as v524
import train_v518_path_pol_vertical_correction as v518
import train_v519_end_to_end_path_spectral_renderer as v519
import train_v525_carrier_residual_renderer_pilot as v525
from project_paths import PATHS


VERSION_NUMBER = 526
VERSION = "v526_full_carrier_residual_renderer"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
FOLD_DIR = CACHE_DIR / "fold_results"
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = CACHE_DIR / "v526_Round1_Test_Channel.npy"
ACTION_CACHE = CACHE_DIR / "v526_path_pdp_actions.npz"
INFERENCE_BUNDLE = CACHE_DIR / "v526_inference_bundle.pt"
ARCHIVE = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

SOURCE_DATA = PATHS.cache_dir(522) / "v522_source_path_coefficients_2500.npz"
BASE = PATHS.cache_dir(491) / "v491_Round1_Test_Channel.npy"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
REGIME_NAMES = v524.REGIME_NAMES
DEPLOYMENT_REGIMES = (1, 2)
TRAINING_REGIMES: tuple[int, ...] | None = None
ENTRY_SCRIPT = "generate_v526_full_carrier_residual_renderer.py"
TINY = 1e-20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max-log-gain", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=526_001)
    return parser.parse_args()


def old_target_for_fold(
    fold: int,
    fold_codes: np.ndarray,
    regimes: np.ndarray,
) -> np.ndarray:
    truth = np.load(v447.TRUTH_CACHE, mmap_mode="r")
    source = np.load(v447.SOURCE_CACHE, mmap_mode="r")
    fit_ids = np.flatnonzero(fold_codes != fold).astype(np.int64)
    regime_names = np.asarray(REGIME_NAMES)[regimes]
    output = np.empty((len(source), 1024, 192), dtype=np.float16)
    for start in range(0, source.shape[1], 16):
        end = min(start + 16, source.shape[1])
        ratios = v447.fit_ratio_chunk(
            truth, source, fit_ids, regime_names, start, end
        )
        values = np.asarray(source[:, start:end], dtype=np.float32)
        local = np.empty_like(values)
        for code, name in enumerate(REGIME_NAMES):
            mask = regimes == code
            if np.any(mask):
                local[mask] = v524.normalize_links(
                    values[mask] * np.exp(ratios[name])[None]
                )
        output[:, start:end] = local.astype(np.float16)
    return output.reshape(len(source), 256, 4, 192)


def features_all(
    data: dict[str, np.ndarray], regimes: np.ndarray
) -> np.ndarray:
    base = v523.source_base_features(data, regimes)
    return v518.input_features(
        base, data["source_coefficients"].astype(np.complex128)
    )


def batch_tensors(
    ids: np.ndarray,
    data: dict[str, np.ndarray],
    features: np.ndarray,
    old: np.ndarray,
    truth: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(features[ids]).to(device),
        torch.from_numpy(data["valid"][ids]).to(device),
        torch.from_numpy(data["tau"][ids].astype(np.float32)).to(device),
        torch.from_numpy(data["h_center"][ids].astype(np.float32)).to(device),
        torch.from_numpy(data["source_coefficients"][ids]).to(device),
        torch.from_numpy(np.asarray(old[ids], dtype=np.float32)).to(device),
        torch.from_numpy(np.asarray(truth[ids], dtype=np.float32)).to(device),
    )


@torch.no_grad()
def evaluate(
    model: v519.PathGainTransformer,
    ids: np.ndarray,
    data: dict[str, np.ndarray],
    features: np.ndarray,
    old: np.ndarray,
    truth: np.ndarray,
    device: torch.device,
    batch_size: int,
    shuffled_ids: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    values = []
    for start in range(0, len(ids), batch_size):
        local = ids[start : start + batch_size]
        tensors = batch_tensors(local, data, features, old, truth, device)
        gain_features = tensors[0]
        gain_valid = tensors[1]
        if shuffled_ids is not None:
            geometry = shuffled_ids[start : start + len(local)]
            gain_features = torch.from_numpy(features[geometry]).to(device)
            gain_valid = torch.from_numpy(data["valid"][geometry]).to(device)
        gains = model.gains(gain_features, gain_valid)
        zero = torch.zeros_like(gains)
        source = v525.normalized_link_pdp(
            v525.render_from_gains(tensors[2], tensors[3], tensors[4], zero)
        )
        corrected = v525.normalized_link_pdp(
            v525.render_from_gains(tensors[2], tensors[3], tensors[4], gains)
        )
        candidate = v525.corrected_carrier_pdp(tensors[5], source, corrected)
        values.extend(v525.point_score(candidate, tensors[6]).cpu().numpy().tolist())
    return np.asarray(values, dtype=np.float32)


def train_folds(
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    raw_features: np.ndarray,
    regimes: np.ndarray,
    fold_codes: np.ndarray,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device(args.device)
    truth = np.load(v447.TRUTH_CACHE, mmap_mode="r").reshape(2000, 256, 4, 192)
    baseline = np.zeros(2000, dtype=np.float32)
    normal = np.zeros(2000, dtype=np.float32)
    shuffled = np.zeros(2000, dtype=np.float32)
    rows = []
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for fold in sorted(np.unique(fold_codes).tolist()):
        fit_ids = np.flatnonzero(fold_codes != fold).astype(np.int64)
        validation_ids = np.flatnonzero(fold_codes == fold).astype(np.int64)
        train_ids = (
            fit_ids
            if TRAINING_REGIMES is None
            else fit_ids[np.isin(regimes[fit_ids], TRAINING_REGIMES)]
        )
        fold_cache = FOLD_DIR / f"fold_{int(fold)}.npz"
        checkpoint_path = CHECKPOINT_DIR / f"fold_{int(fold)}.pt"
        if fold_cache.is_file() and checkpoint_path.is_file():
            with np.load(fold_cache) as loaded:
                saved_ids = loaded["validation_ids"].astype(np.int64)
                if not np.array_equal(saved_ids, validation_ids):
                    raise ValueError(f"fold cache mismatch: {fold_cache}")
                cached_detail = json.loads(str(loaded["details"].item()))
                cached_values = [
                    cached_detail["baseline"],
                    cached_detail["normal"],
                    cached_detail["shuffled"],
                    cached_detail["gain"],
                    cached_detail["margin_vs_shuffled"],
                    *[
                        value
                        for row in cached_detail["history"]
                        for value in row.values()
                        if isinstance(value, (int, float))
                    ],
                ]
                cache_finite = bool(np.isfinite(cached_values).all())
                if cache_finite:
                    baseline[validation_ids] = loaded["baseline"]
                    normal[validation_ids] = loaded["normal"]
                    shuffled[validation_ids] = loaded["shuffled"]
                    rows.append(cached_detail)
            if cache_finite:
                print(f"reuse v526 fold {fold}", flush=True)
                continue
            print(f"discard non-finite v526 fold {fold} cache", flush=True)

        old = old_target_for_fold(int(fold), fold_codes, regimes)
        for start in range(0, len(validation_ids), 16):
            ids = validation_ids[start : start + 16]
            truth_local = np.asarray(truth[ids], dtype=np.float32).reshape(len(ids), 1024, 192)
            old_local = np.asarray(old[ids], dtype=np.float32).reshape(len(ids), 1024, 192)
            baseline[ids] = np.mean(v524.link_cosine(truth_local, old_local), axis=1)

        selected_features = raw_features[train_ids][data["valid"][train_ids]]
        mean = selected_features.mean(axis=0, keepdims=True).astype(np.float32)
        std = np.maximum(selected_features.std(axis=0, keepdims=True), 1e-4).astype(np.float32)
        features = ((raw_features - mean) / std).astype(np.float32)
        features *= data["valid"][..., None]
        torch.manual_seed(args.seed + int(fold))
        rng = np.random.default_rng(args.seed + int(fold))
        model = v519.PathGainTransformer(features.shape[2], args.max_log_gain).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        validation_mask = np.isin(regimes[validation_ids], DEPLOYMENT_REGIMES)
        best_score = float(np.mean(baseline[validation_ids][validation_mask]))
        best_epoch = 0
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        stale = 0
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = rng.permutation(train_ids)
            running = 0.0
            for start in range(0, len(order), args.batch_size):
                ids = order[start : start + args.batch_size]
                tensors = batch_tensors(ids, data, features, old, truth, device)
                gains = model.gains(tensors[0], tensors[1])
                zero = torch.zeros_like(gains)
                source = v525.normalized_link_pdp(
                    v525.render_from_gains(tensors[2], tensors[3], tensors[4], zero)
                )
                corrected = v525.normalized_link_pdp(
                    v525.render_from_gains(tensors[2], tensors[3], tensors[4], gains)
                )
                candidate = v525.corrected_carrier_pdp(tensors[5], source, corrected)
                score = v525.point_score(candidate, tensors[6])
                loss = 1.0 - torch.mean(score) + 1e-4 * torch.mean(gains * gains)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running += float(loss.detach().cpu()) * len(ids)
            if epoch == 1 or epoch % 2 == 0:
                values = evaluate(
                    model, validation_ids, data, features, old, truth,
                    device, args.batch_size
                )
                score = float(np.mean(values[validation_mask]))
                history.append(
                    {"epoch": epoch, "train_loss": running / len(train_ids), "validation_light_medium": score}
                )
                if score > best_score + 1e-5:
                    best_score = score
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 2
                    if stale >= args.patience:
                        break
        model.load_state_dict(best_state)
        normal[validation_ids] = evaluate(
            model, validation_ids, data, features, old, truth,
            device, args.batch_size
        )
        shuffled_ids = validation_ids[
            np.random.default_rng(args.seed + 10_000 + int(fold)).permutation(len(validation_ids))
        ]
        shuffled[validation_ids] = evaluate(
            model, validation_ids, data, features, old, truth,
            device, args.batch_size, shuffled_ids=shuffled_ids
        )
        deployed = validation_mask
        fold_detail = {
            "fold": int(fold),
            "fit_count": int(len(fit_ids)),
            "train_count": int(len(train_ids)),
            "validation_count": int(len(validation_ids)),
            "deployed_validation_count": int(np.sum(deployed)),
            "best_epoch": int(best_epoch),
            "baseline": float(np.mean(baseline[validation_ids][deployed])),
            "normal": float(np.mean(normal[validation_ids][deployed])),
            "shuffled": float(np.mean(shuffled[validation_ids][deployed])),
            "gain": float(np.mean(normal[validation_ids][deployed] - baseline[validation_ids][deployed])),
            "margin_vs_shuffled": float(np.mean(normal[validation_ids][deployed] - shuffled[validation_ids][deployed])),
            "history": history,
        }
        rows.append(fold_detail)
        torch.save(
            {
                "state_dict": best_state,
                "feature_mean": mean,
                "feature_std": std,
                "input_dim": int(features.shape[2]),
                "max_log_gain": float(args.max_log_gain),
                "fold": int(fold),
                "best_epoch": int(best_epoch),
            },
            checkpoint_path,
        )
        np.savez_compressed(
            fold_cache,
            validation_ids=validation_ids,
            baseline=baseline[validation_ids],
            normal=normal[validation_ids],
            shuffled=shuffled[validation_ids],
            details=np.asarray(json.dumps(fold_detail)),
        )
        print(json.dumps(fold_detail), flush=True)
        del model, old
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows, baseline, normal, shuffled


@torch.no_grad()
def ensemble_test_log_ratio(
    data: dict[str, np.ndarray], raw_features: np.ndarray, device: torch.device
) -> tuple[np.ndarray, list[dict[str, object]]]:
    accumulator = np.zeros((500, 256, 4, 192), dtype=np.float32)
    model_rows = []
    checkpoint_paths = sorted(CHECKPOINT_DIR.glob("fold_*.pt"))
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = v519.PathGainTransformer(
            int(checkpoint["input_dim"]), float(checkpoint["max_log_gain"])
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        features = (
            (raw_features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
        ).astype(np.float32)
        features *= data["valid"][..., None]
        for start in range(2000, 2500, 8):
            end = min(start + 8, 2500)
            ids = np.arange(start, end)
            local_features = torch.from_numpy(features[ids]).to(device)
            valid = torch.from_numpy(data["valid"][ids]).to(device)
            gains = model.gains(local_features, valid)
            zero = torch.zeros_like(gains)
            tau = torch.from_numpy(data["tau"][ids].astype(np.float32)).to(device)
            h_center = torch.from_numpy(data["h_center"][ids].astype(np.float32)).to(device)
            source_coefficients = torch.from_numpy(data["source_coefficients"][ids]).to(device)
            source = v525.normalized_link_pdp(
                v525.render_from_gains(tau, h_center, source_coefficients, zero)
            )
            corrected = v525.normalized_link_pdp(
                v525.render_from_gains(tau, h_center, source_coefficients, gains)
            )
            floor = 1e-6 / 192.0
            ratio = torch.clamp(
                torch.log(torch.clamp(corrected, min=floor))
                - torch.log(torch.clamp(source, min=floor)),
                -2.0,
                2.0,
            )
            accumulator[start - 2000 : end - 2000] += ratio.cpu().numpy()
        model_rows.append(
            {"fold": int(checkpoint["fold"]), "best_epoch": int(checkpoint["best_epoch"])}
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    accumulator /= max(len(checkpoint_paths), 1)
    return accumulator, model_rows


def write_candidate(log_ratio: np.ndarray, test_regimes: np.ndarray) -> dict[str, object]:
    base = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    action = np.zeros(500, dtype=np.float32)
    power_ratio = np.zeros(500, dtype=np.float64)
    for start in range(0, 500, 2):
        end = min(start + 2, 500)
        channel = np.asarray(base[start:end], dtype=np.complex64)
        cir = np.fft.ifft(channel, axis=-1).astype(np.complex64)
        current_power = np.abs(cir).astype(np.float32) ** 2
        energy = np.sum(current_power, axis=-1, keepdims=True, dtype=np.float64)
        current = current_power / np.maximum(energy, TINY)
        active = np.isin(test_regimes[start:end], DEPLOYMENT_REGIMES).astype(np.float32)
        target = v524.normalize_links(
            current * np.exp(active[:, None, None, None] * log_ratio[start:end])
        )
        action[start:end] = np.mean(
            np.sum(np.abs(target - current), axis=-1), axis=(1, 2)
        )
        phase = cir / np.maximum(np.abs(cir), TINY)
        new_cir = phase * np.sqrt(target * energy).astype(np.float32)
        prediction = np.fft.fft(new_cir, axis=-1).astype(np.complex64)
        output[start:end] = prediction
        old_power = np.sum(np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64)
        new_power = np.sum(np.abs(prediction) ** 2, axis=(1, 2, 3), dtype=np.float64)
        power_ratio[start:end] = new_power / np.maximum(old_power, TINY)
    output.flush()
    np.savez_compressed(
        ACTION_CACHE,
        test_regimes=test_regimes,
        deployment_regimes=np.asarray(DEPLOYMENT_REGIMES),
        action_l1=action,
    )
    return {
        "active_points": int(np.sum(np.isin(test_regimes, DEPLOYMENT_REGIMES))),
        "action_l1_mean": float(np.mean(action)),
        "action_l1_by_regime": {
            name: float(np.mean(action[test_regimes == code]))
            for code, name in enumerate(REGIME_NAMES)
        },
        "power_ratio_mean": float(np.mean(power_ratio)),
        "power_ratio_max_abs_error": float(np.max(np.abs(power_ratio - 1.0))),
    }


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def package() -> dict[str, object]:
    dependencies = (
        ENTRY_SCRIPT,
        "generate_v526_full_carrier_residual_renderer.py",
        "train_v525_carrier_residual_renderer_pilot.py",
        "generate_v524_carrier_aware_path_pdp_residual.py",
        "generate_v523_end_to_end_path_pdp_residual.py",
        "train_v519_end_to_end_path_spectral_renderer.py",
        "train_v518_path_pol_vertical_correction.py",
        "project_paths.py",
    )
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for name in dependencies:
            archive.write(PATHS.code / name, name)
        archive.write(INFERENCE_BUNDLE, "model/v526_inference_bundle.pt")
        archive.write(SOURCE_DATA, "model/v522_source_path_coefficients_2500.npz")
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"bad ZIP member: {bad}")
    return {
        "zip_crc": True,
        "zip_members": members,
        "zip_size": int(ARCHIVE.stat().st_size),
        "sha256": digest(ARCHIVE, "sha256"),
        "md5": digest(ARCHIVE, "md5"),
    }


def main() -> None:
    args = parse_args()
    for path in (SOURCE_DATA, BASE, TRAIN_POS, v447.TRUTH_CACHE, v447.SOURCE_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE_DATA) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}
    regimes = v524.regimes_all()
    positions = np.load(TRAIN_POS).astype(np.float32)
    fold_codes = v501.spatial_fold_codes(positions, regimes[:2000])
    raw_features = features_all(data, regimes)
    rows, baseline, normal, shuffled = train_folds(
        args, data, raw_features, regimes[:2000], fold_codes
    )
    deployed = np.isin(regimes[:2000], DEPLOYMENT_REGIMES)
    deployed_normal = np.where(deployed, normal, baseline)
    deployed_shuffled = np.where(deployed, shuffled, baseline)
    gain = deployed_normal - baseline
    margin = deployed_normal - deployed_shuffled
    positive_folds = int(sum(row["gain"] > 1e-5 for row in rows))
    positive_margin_folds = int(sum(row["margin_vs_shuffled"] > 1e-5 for row in rows))
    passed = bool(
        np.mean(gain) >= 0.003
        and positive_folds >= 4
        and np.mean(margin) >= 0.003
        and positive_margin_folds >= 4
    )
    oof = {
        "deployment_regimes": [REGIME_NAMES[code] for code in DEPLOYMENT_REGIMES],
        "training_regimes": (
            list(REGIME_NAMES)
            if TRAINING_REGIMES is None
            else [REGIME_NAMES[code] for code in TRAINING_REGIMES]
        ),
        "baseline": float(np.mean(baseline)),
        "deployed": float(np.mean(deployed_normal)),
        "gain": float(np.mean(gain)),
        "margin_vs_shuffled": float(np.mean(margin)),
        "positive_folds": positive_folds,
        "positive_margin_folds": positive_margin_folds,
        "folds": rows,
    }
    if not passed:
        report = {
            "method": VERSION,
            "status": "analysis_rejected_no_candidate",
            "primary_hypothesis": "Full-scale carrier-aware path residual learning transfers beyond the 160-point v525 pilot.",
            "oof": oof,
            "deployment_gate": {"pass": False, "minimum_gain": 0.003, "minimum_positive_folds": 4, "minimum_margin": 0.003, "minimum_positive_margin_folds": 4},
        }
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return

    device = torch.device(args.device)
    log_ratio, model_rows = ensemble_test_log_ratio(data, raw_features, device)
    action = write_candidate(log_ratio, regimes[2000:])
    bundle = {
        "models": [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in sorted(CHECKPOINT_DIR.glob("fold_*.pt"))
        ],
        "deployment_regimes": DEPLOYMENT_REGIMES,
        "parameters": vars(args),
    }
    torch.save(bundle, INFERENCE_BUNDLE)
    value = np.load(OUTPUT, mmap_mode="r")
    integrity = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(
            all(
                np.isfinite(np.asarray(value[start : start + 8])).all()
                for start in range(0, len(value), 8)
            )
        ),
    }
    integrity.update(package())
    report = {
        "method": VERSION,
        "status": "candidate_pending_diagnostic_noncompliant",
        "primary_hypothesis": "Full-scale carrier-aware path residual learning transfers beyond the 160-point v525 pilot.",
        "base": {"version": 491, "online_score": 0.739634},
        "compliance_boundary": {
            "diagnostic_only": True,
            "explicitly_noncompliant_for_final_submission": True,
            "reason": "uses Sionna path coefficients generated at official test coordinates",
        },
        "oof": oof,
        "deployment_gate": {"pass": True, "minimum_gain": 0.003, "minimum_positive_folds": 4, "minimum_margin": 0.003, "minimum_positive_margin_folds": 4},
        "ensemble": model_rows,
        "action": action,
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTION_CACHE),
            "inference_bundle": PATHS.relative(INFERENCE_BUNDLE),
            "archive": PATHS.relative(ARCHIVE),
            "report": PATHS.relative(REPORT),
        },
        "integrity": integrity,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
