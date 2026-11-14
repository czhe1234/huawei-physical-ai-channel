#!/usr/bin/env python3
"""v237 analysis: low-rank complex-channel denoising for path-PDP learning.

Released train H is modeled as a low-rank multipath matrix plus complex white
Gaussian noise.  Batched SVD shrinkage is selected by independent antenna-link
half consistency, then the v236 path-PDP decoder is retrained on the selected
denoised target.  The local scorer remains evaluation-only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import analyze_v233_official_joint_ad_manifold as v233
import analyze_v236_official_path_pdp_decoder as v236
import generate_v215_pointcloud_physics_neural_field as v215
import generate_v218_official_multispectrum_physics_ai as v218
from generate_v87_easy_field_projection import summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 237
VERSION = "v237_lowrank_denoised_path_pdp"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v237_lowrank_denoised_path_pdp.json"
ACTION_OUT = OUT_DIR / "v237_lowrank_denoised_actions.npz"
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

FACTORS = (0.78, 0.94, 1.10)
DENOISE_BATCH = 4
GATE_SCORE = 0.645604
BASE_V236 = PATHS.cache_dir(236) / "v236_path_selective_Round1_Test_Channel.npy"


def normalize_d(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 1e-12)
    out /= np.maximum(out.sum(axis=1, keepdims=True), TINY)
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = a.reshape(len(a), -1).astype(np.float64)
    y = b.reshape(len(b), -1).astype(np.float64)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def entropy_rows(x: np.ndarray) -> np.ndarray:
    value = normalize_d(x)
    return (-np.sum(value * np.log(np.maximum(value, 1e-12)), axis=1) / math.log(value.shape[1])).astype(np.float32)


def denoised_cache(factor: float) -> Path:
    return OUT_DIR / f"v237_train_denoised_d_f{factor:.2f}.npy"


def reconstruct_d(
    u: torch.Tensor,
    s: torch.Tensor,
    vh: torch.Tensor,
    sigma: torch.Tensor,
    rows: int,
    factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    threshold = sigma * (math.sqrt(rows) + math.sqrt(vh.shape[-1])) * float(factor)
    keep = s > threshold[:, None]
    debias = sigma[:, None] ** 2 * float(rows + vh.shape[-1])
    shrink = torch.sqrt(torch.clamp(s**2 - debias, min=0.0)) * keep
    recon = torch.matmul(u * shrink[:, None, :], vh)
    d = torch.sum(torch.abs(recon) ** 2, dim=1).float()
    d /= d.sum(dim=1, keepdim=True).clamp_min(1e-30)
    return d, keep.sum(dim=1).float()


def svd_parts(cir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    late = torch.abs(cir[..., 64:]) ** 2
    sigma2 = torch.quantile(late.reshape(len(cir), -1), 0.5, dim=1) / math.log(2.0)
    sigma = torch.sqrt(torch.clamp(sigma2, min=0.0))
    u, s, vh = torch.linalg.svd(cir, full_matrices=False)
    return u, s, vh, sigma


def build_denoised_targets(device: torch.device) -> tuple[dict[float, np.ndarray], dict[str, object]]:
    paths = {factor: denoised_cache(factor) for factor in FACTORS}
    diag_path = OUT_DIR / "v237_denoise_diagnostics.npz"
    if all(path.is_file() for path in paths.values()) and diag_path.is_file():
        diag = np.load(diag_path)
        report = {
            "raw_half_cos": summarize(diag["raw_half_cos"]),
            "factors": {
                f"{factor:.2f}": {
                    "half_cos": summarize(diag[f"half_cos_{factor:.2f}"]),
                    "full_vs_raw": summarize(diag[f"full_raw_{factor:.2f}"]),
                    "rank": summarize(diag[f"rank_{factor:.2f}"]),
                    "entropy": summarize(diag[f"entropy_{factor:.2f}"]),
                }
                for factor in FACTORS
            },
        }
        return {factor: np.load(path, mmap_mode="r") for factor, path in paths.items()}, report

    train_h = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    outputs = {
        factor: np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(train_h), train_h.shape[-1]))
        for factor, path in paths.items()
    }
    raw_half_cos = np.zeros(len(train_h), dtype=np.float32)
    half_cos = {factor: np.zeros(len(train_h), dtype=np.float32) for factor in FACTORS}
    full_raw = {factor: np.zeros(len(train_h), dtype=np.float32) for factor in FACTORS}
    ranks = {factor: np.zeros(len(train_h), dtype=np.float32) for factor in FACTORS}
    entropies = {factor: np.zeros(len(train_h), dtype=np.float32) for factor in FACTORS}

    for start in range(0, len(train_h), DENOISE_BATCH):
        end = min(start + DENOISE_BATCH, len(train_h))
        h = np.asarray(train_h[start:end], dtype=np.complex64).reshape(end - start, -1, train_h.shape[-1])
        cir_np = np.fft.ifft(h, axis=-1).astype(np.complex64)
        raw_full = normalize_d(np.sum(np.abs(cir_np) ** 2, axis=1).astype(np.float32))
        raw_even = normalize_d(np.sum(np.abs(cir_np[:, 0::2]) ** 2, axis=1).astype(np.float32))
        raw_odd = normalize_d(np.sum(np.abs(cir_np[:, 1::2]) ** 2, axis=1).astype(np.float32))
        raw_half_cos[start:end] = row_cos(raw_even, raw_odd)

        full = torch.from_numpy(cir_np).to(device)
        even = full[:, 0::2]
        odd = full[:, 1::2]
        full_parts = svd_parts(full)
        even_parts = svd_parts(even)
        odd_parts = svd_parts(odd)
        for factor in FACTORS:
            d_full, rank = reconstruct_d(*full_parts, full.shape[1], factor)
            d_even, _ = reconstruct_d(*even_parts, even.shape[1], factor)
            d_odd, _ = reconstruct_d(*odd_parts, odd.shape[1], factor)
            full_np = d_full.cpu().numpy().astype(np.float32)
            outputs[factor][start:end] = full_np
            outputs[factor].flush()
            half_cos[factor][start:end] = row_cos(d_even.cpu().numpy(), d_odd.cpu().numpy())
            full_raw[factor][start:end] = row_cos(full_np, raw_full)
            ranks[factor][start:end] = rank.cpu().numpy()
            entropies[factor][start:end] = entropy_rows(full_np)
        print(f"v237 SVD denoise {end}/{len(train_h)}", flush=True)

    payload = {"raw_half_cos": raw_half_cos}
    for factor in FACTORS:
        payload[f"half_cos_{factor:.2f}"] = half_cos[factor]
        payload[f"full_raw_{factor:.2f}"] = full_raw[factor]
        payload[f"rank_{factor:.2f}"] = ranks[factor]
        payload[f"entropy_{factor:.2f}"] = entropies[factor]
    np.savez_compressed(diag_path, **payload)
    report = {
        "raw_half_cos": summarize(raw_half_cos),
        "factors": {
            f"{factor:.2f}": {
                "half_cos": summarize(half_cos[factor]),
                "full_vs_raw": summarize(full_raw[factor]),
                "rank": summarize(ranks[factor]),
                "entropy": summarize(entropies[factor]),
            }
            for factor in FACTORS
        },
    }
    return {factor: np.load(path, mmap_mode="r") for factor, path in paths.items()}, report


def select_factor(report: dict[str, object]) -> float:
    scored = []
    for factor in FACTORS:
        item = report["factors"][f"{factor:.2f}"]
        score = float(item["half_cos"]["mean"]) + 0.18 * float(item["full_vs_raw"]["mean"])
        scored.append((score, factor))
    return max(scored)[1]


def augment_raw(
    raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    anchor_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global_features, path_numeric, path_type, path_valid = raw
    delay_features = np.concatenate(
        [np.log(np.maximum(anchor_d, 1e-10)), v215.d_stats(anchor_d)], axis=1
    ).astype(np.float32)
    return (
        np.concatenate([global_features, delay_features], axis=1).astype(np.float32),
        path_numeric,
        path_type,
        path_valid,
    )


def run_variant(
    name: str,
    source: Path,
    predicted_d: np.ndarray,
    alpha: np.ndarray,
    regimes: np.ndarray,
    clip: float,
) -> dict[str, object]:
    current = v225.current_delay_marginals(source)
    target_d = normalize_d((1.0 - alpha[:, None]) * current["d"] + alpha[:, None] * predicted_d)
    targets = {"d": target_d, "du": current["du"], "dh": current["dh"]}
    output = OUT_DIR / f"v237_{name}_Round1_Test_Channel.npy"
    diag = v225.apply_delay_targets(source, output, targets, regimes, clip)
    return {
        "name": name,
        "source": PATHS.relative(source),
        "output": PATHS.relative(output),
        "score": v208.score_prediction(output),
        "alpha": summarize(alpha),
        "target_l1": summarize(v225.l1_rows(target_d, current["d"])),
        "diagnostics": {key: summarize(value) for key, value in diag.items()},
    }


def main() -> None:
    if not BASE_V236.is_file():
        raise FileNotFoundError(BASE_V236)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets_by_factor, denoise_report = build_denoised_targets(device)
    factor = select_factor(denoise_report)
    denoised_all = normalize_d(np.asarray(targets_by_factor[factor], dtype=np.float32))

    support, valid_joint, log_power = v233.load_normalized_train_support()
    raw_total = np.load(PATHS.shared_features / "train_total_power.npy")
    valid = valid_joint & (raw_total > 1e-20)
    valid_ids = np.flatnonzero(valid).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(v233.GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    bundle = v233.make_anchor_bundle(support, valid_ids, log_power, train_pos, test_pos, train_blocked, test_blocked)
    train_ids = bundle["train_ids"]
    val_ids = bundle["val_ids"]

    idx_train, w_train = v218.neighbor_index_weights(
        train_pos[train_ids], train_blocked[train_ids], train_pos[train_ids], train_blocked[train_ids],
        query_ids=train_ids, bank_ids=train_ids, k=v233.ANCHOR_K,
    )
    idx_val, w_val = v218.neighbor_index_weights(
        train_pos[val_ids], train_blocked[val_ids], train_pos[train_ids], train_blocked[train_ids], k=v233.ANCHOR_K,
    )
    train_anchor_d = v218.weighted_average(denoised_all[train_ids], idx_train, w_train)
    val_anchor_d = v218.weighted_average(denoised_all[train_ids], idx_val, w_val)
    train_raw = augment_raw(v233.raw_features(
        "train", train_ids, train_pos[train_ids], train_blocked[train_ids], bundle["train_anchor"], bundle["train_anchor_power"], bundle["w_train"]
    ), train_anchor_d)
    val_raw = augment_raw(v233.raw_features(
        "train", val_ids, train_pos[val_ids], train_blocked[val_ids], bundle["val_anchor"], bundle["val_anchor_power"], bundle["w_val"]
    ), val_anchor_d)
    train_features, [val_features], _ = v233.standardize_feature_sets(train_raw, [val_raw])
    train_delay = v236.raw_path_delay("train", train_ids)
    val_delay = v236.raw_path_delay("train", val_ids)
    model, validation_report = v236.train_model(
        train_features, train_delay, train_anchor_d, denoised_all[train_ids],
        v236.SEEDS[0], v236.TRAIN_EPOCHS, device,
        (val_features, val_delay, val_anchor_d, denoised_all[val_ids]),
    )
    val_out = v236.predict_model(model, val_features, val_delay, val_anchor_d, device)
    val_alpha, calibration_report = v236.calibrate_alpha(
        val_anchor_d, val_out.d, denoised_all[val_ids], regime_of(train_blocked[val_ids])
    )
    validation_metrics = {
        "anchor_cos": float(np.mean(row_cos(val_anchor_d, denoised_all[val_ids]))),
        "model_cos": float(np.mean(row_cos(val_out.d, denoised_all[val_ids]))),
        "path_cos": float(np.mean(row_cos(val_out.path_d, denoised_all[val_ids]))),
        "path_mass": summarize(val_out.path_mass),
        "tail_mass": summarize(val_out.tail_mass),
    }
    best_epoch = max(10, int(validation_report["best_epoch"]))
    del model, val_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    idx_final, w_final = v218.neighbor_index_weights(
        train_pos[valid_ids], train_blocked[valid_ids], train_pos[valid_ids], train_blocked[valid_ids],
        query_ids=valid_ids, bank_ids=valid_ids, k=v233.ANCHOR_K,
    )
    idx_test, w_test = v218.neighbor_index_weights(
        test_pos, test_blocked, train_pos[valid_ids], train_blocked[valid_ids], k=v233.ANCHOR_K,
    )
    final_anchor_d = v218.weighted_average(denoised_all[valid_ids], idx_final, w_final)
    test_anchor_d = v218.weighted_average(denoised_all[valid_ids], idx_test, w_test)
    final_raw = augment_raw(v233.raw_features(
        "train", valid_ids, train_pos[valid_ids], train_blocked[valid_ids], bundle["final_anchor"], bundle["final_anchor_power"], bundle["w_final"]
    ), final_anchor_d)
    test_raw = augment_raw(v233.raw_features(
        "test", None, test_pos, test_blocked, bundle["test_anchor"], bundle["test_anchor_power"], bundle["w_test"]
    ), test_anchor_d)
    final_features, [test_features], final_stats = v233.standardize_feature_sets(final_raw, [test_raw])
    final_delay = v236.raw_path_delay("train", valid_ids)
    test_delay = v236.raw_path_delay("test", None)
    seed_outputs = []
    final_training = []
    for seed in v236.SEEDS:
        final_model, info = v236.train_model(
            final_features, final_delay, final_anchor_d, denoised_all[valid_ids], seed, best_epoch, device, None
        )
        torch.save(
            {"state_dict": final_model.state_dict(), "feature_stats": final_stats, "seed": seed, "epochs": best_epoch, "denoise_factor": factor},
            CHECKPOINT_DIR / f"seed_{seed}.pt",
        )
        seed_outputs.append(v236.predict_model(final_model, test_features, test_delay, test_anchor_d, device))
        final_training.append(info)
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    test_d = normalize_d(np.mean(np.stack([item.d for item in seed_outputs]), axis=0))
    test_path_d = normalize_d(np.mean(np.stack([item.path_d for item in seed_outputs]), axis=0))
    agreement = np.mean(np.stack([
        row_cos(seed_outputs[i].d, seed_outputs[j].d)
        for i in range(len(seed_outputs)) for j in range(i + 1, len(seed_outputs))
    ]), axis=0).astype(np.float32)
    confidence = np.clip((agreement - 0.94) / 0.055, 0.0, 1.0).astype(np.float32)
    regimes = regime_of(test_blocked)
    alpha_base = np.asarray([val_alpha[str(regime)] for regime in regimes], dtype=np.float32)
    variants = [
        ("path_on_v236", BASE_V236, test_path_d,
         np.clip(np.asarray([{"clear": 0.0, "light": 0.04, "medium": 0.20, "heavy": 0.34}[str(r)] for r in regimes], dtype=np.float32) * confidence, 0.0, 0.38), 1.02),
        ("model_on_v236", BASE_V236, test_d,
         np.clip(alpha_base * (0.38 + 0.62 * confidence), 0.0, 0.86), 1.02),
        ("path_on_v226", v233.BASE, test_path_d,
         np.clip(np.asarray([{"clear": 0.0, "light": 0.05, "medium": 0.25, "heavy": 0.42}[str(r)] for r in regimes], dtype=np.float32) * confidence, 0.0, 0.45), 1.02),
        ("model_aggressive_on_v236", BASE_V236, test_d,
         np.clip(1.35 * alpha_base * (0.42 + 0.58 * confidence), 0.0, 1.02), 1.08),
    ]
    results = [run_variant(name, source, pred, alpha.astype(np.float32), regimes, clip) for name, source, pred, alpha, clip in variants]
    best = max(results, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(
        ACTION_OUT,
        denoise_factor=np.asarray(factor, dtype=np.float32),
        agreement=agreement,
        confidence=confidence,
        model_vs_anchor=row_cos(test_d, test_anchor_d),
        path_vs_v236=row_cos(test_path_d, v225.current_delay_marginals(BASE_V236)["d"]),
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": "Low-rank multipath plus complex white Gaussian noise permits self-supervised SVD shrinkage to recover cleaner official PDP supervision.",
        "compliance_boundary": {
            "generation_uses": [
                "official train H",
                "official train/test coordinates",
                "official point-cloud features and path candidates",
                "v236/v226 strict official-data carriers",
            ],
            "generation_does_not_use": [
                "Sionna generated channel labels",
                "v212/v217/v221/v222 teacher outputs",
                "synthetic replay truth",
                "hidden online labels",
                "raw RT H",
            ],
        },
        "denoising": {"factors": list(FACTORS), "selected_factor": factor, "diagnostics": denoise_report},
        "training": {
            "device": str(device),
            "best_epoch": best_epoch,
            "validation": validation_report,
            "validation_metrics": validation_metrics,
            "calibration": calibration_report,
            "alpha_by_regime": val_alpha,
            "final_training": final_training,
        },
        "test_diagnostics": {
            "agreement": summarize(agreement),
            "confidence": summarize(confidence),
            "model_vs_anchor": summarize(row_cos(test_d, test_anchor_d)),
        },
        "variants": results,
        "local_gate": {
            "required_score_traceback_leaked": GATE_SCORE,
            "best_variant": best["name"],
            "best_score_traceback_leaked": best["score"]["score_traceback_leaked"],
            "gap_to_gate": best["score"]["score_traceback_leaked"] - GATE_SCORE,
        },
        "outputs": {"actions": PATHS.relative(ACTION_OUT), "report": PATHS.relative(REPORT)},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
