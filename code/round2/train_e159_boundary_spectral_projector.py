#!/usr/bin/env python3
"""Predict boundary-conditioned spectra and project them onto the V7 carrier."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import analyze_e153_v7_v12_peak_actions as e153
import train_e158_full_boundary_wavefield_operator as e158
from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    fixed_blend,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
)
from data import load_positions
from official_metrics import aggregate_statistics, point_metric_statistics
from project_paths import PATHS
from run_foundation_benchmark import POWER_CACHE
from run_rbf_spectral_field import FOLD_CACHE, OOF_META


VERSION = "Round2-E159-boundary-spectral-projector"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "boundary_spectral_projector"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_boundary_spectral_projector"
REPORT = ANALYSIS_DIR / "boundary_spectral_projector.json"

TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
MAP_FEATURES = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)

OUTER = 3
ALPHA_PAS = 0.20
ALPHA_PDP = 0.20
PROJECTION_ROUNDS = 4
REQUIRED_SCORE_GAIN = 0.005
MAX_NMSE_DEGRADATION = 0.01
TINY = 1e-30


def power_cosine(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    vector_dims: tuple[int, ...],
) -> torch.Tensor:
    numerator = torch.sum(prediction * truth, dim=vector_dims)
    prediction_norm = torch.sum(prediction.square(), dim=vector_dims)
    truth_norm = torch.sum(truth.square(), dim=vector_dims)
    denominator = torch.sqrt(prediction_norm * truth_norm)
    denominator = torch.where(
        denominator == 0.0,
        torch.full_like(denominator, 1e-12),
        denominator,
    )
    return torch.mean(numerator / denominator)


def spectral_metrics_torch(
    prediction: torch.Tensor, truth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction_pas = torch.fft.fft(prediction, dim=-1, norm="ortho").abs().square()
    truth_pas = torch.fft.fft(truth, dim=-1, norm="ortho").abs().square()
    pas = power_cosine(prediction_pas, truth_pas, (1, 2, 3))

    prediction_pdp = torch.fft.ifft2(
        prediction, dim=(2, 3), norm="ortho"
    ).abs().square()
    truth_pdp = torch.fft.ifft2(truth, dim=(2, 3), norm="ortho").abs().square()
    pdp = power_cosine(prediction_pdp, truth_pdp, (-1,))
    return pas, pdp


def train_spectral_model(
    dataset: e158.WavefieldDataset,
    config: e158.Config,
    point_power: np.ndarray,
    device: torch.device,
) -> tuple[e158.FullBoundaryWavefieldOperator, list[dict[str, float]]]:
    e158.set_seed(config.seed + 159)
    model = e158.FullBoundaryWavefieldOperator(
        dataset.table.query_features.shape[1],
        dataset.table.donor_features.shape[2],
        config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    available = np.flatnonzero(
        point_power[dataset.table.query_ids] > 0.0
    ).astype(np.int64)
    if len(available) == 0:
        raise RuntimeError("no nonzero training rows for E159")
    rng = np.random.default_rng(config.seed + 159)
    history: list[dict[str, float]] = []
    rolling: list[tuple[float, float, float]] = []

    for step in range(1, config.screen_steps + 1):
        rows = rng.choice(
            available,
            size=config.batch_size,
            replace=len(available) < config.batch_size,
        )
        donor_field, donor_features, query_features, target = dataset.batch(
            rows, boundary_enabled=True
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(donor_field, donor_features, query_features).to(
            torch.complex64
        )
        pas, pdp = spectral_metrics_torch(prediction, target)
        loss = 1.0 - 0.5 * (pas + pdp)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite E159 loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        rolling.append(
            (
                float(loss.detach().cpu()),
                float(pas.detach().cpu()),
                float(pdp.detach().cpu()),
            )
        )
        if step % config.log_interval == 0 or step == config.screen_steps:
            record = {
                "step": float(step),
                "loss": float(np.mean([row[0] for row in rolling])),
                "pas": float(np.mean([row[1] for row in rolling])),
                "pdp": float(np.mean([row[2] for row in rolling])),
            }
            history.append(record)
            print(
                f"E159 step={step}/{config.screen_steps} "
                f"PAS={record['pas']:.6f} PDP={record['pdp']:.6f}",
                flush=True,
            )
            rolling.clear()
    return model, history


def load_v7_channel(train_count: int, query_ids: np.ndarray) -> np.ndarray:
    metadata = np.load(OOF_META, allow_pickle=False)
    all_query_ids = metadata["query_ids"].astype(np.int64)
    row_map = np.full(train_count, -1, dtype=np.int64)
    row_map[all_query_ids] = np.arange(len(all_query_ids), dtype=np.int64)
    rows = row_map[query_ids]
    if np.any(rows < 0):
        raise RuntimeError("E159 query missing from V7 OOF channel")
    v7 = np.load(e153.V7_CHANNEL, mmap_mode="r")
    return np.asarray(v7[rows], dtype=np.complex64)


@torch.no_grad()
def project_spectra_onto_v7(
    base: np.ndarray,
    spectral_prediction: np.ndarray,
    device: torch.device,
    batch_size: int = 2,
) -> np.ndarray:
    output = np.empty_like(base, dtype=np.complex64)
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        carrier = torch.as_tensor(
            np.asarray(base[start:end]), dtype=torch.complex64, device=device
        )
        target = torch.as_tensor(
            np.asarray(spectral_prediction[start:end]),
            dtype=torch.complex64,
            device=device,
        )
        target_pas = pas_distribution(target)
        target_pdp = pdp_distribution(target)
        reference_power = carrier.abs().square().sum(dim=(1, 2, 3))

        desired_pas = fixed_blend(
            pas_distribution(carrier), target_pas, ALPHA_PAS, (1, 2, 3)
        )
        candidate = exact_pas_projection(carrier, desired_pas)
        desired_pdp = fixed_blend(
            pdp_distribution(candidate), target_pdp, ALPHA_PDP, (3,)
        )
        candidate = exact_pdp_projection(candidate, desired_pdp)
        candidate = preserve_point_power(candidate, reference_power)

        for _ in range(PROJECTION_ROUNDS - 1):
            candidate = exact_pas_projection(candidate, desired_pas)
            candidate = exact_pdp_projection(candidate, desired_pdp)
            candidate = preserve_point_power(candidate, reference_power)
        output[start:end] = candidate.cpu().numpy().astype(np.complex64)
    return output


def save_checkpoint(
    path: Path,
    model: e158.FullBoundaryWavefieldOperator,
    config: e158.Config,
) -> None:
    torch.save(
        {
            "state_dict": copy.deepcopy(model.state_dict()),
            "config": asdict(config),
            "outer": OUTER,
            "alpha_pas": ALPHA_PAS,
            "alpha_pdp": ALPHA_PDP,
            "projection_rounds": PROJECTION_ROUNDS,
        },
        path,
    )


def main() -> None:
    started = time.time()
    config = e158.Config()
    e158.set_seed(config.seed + 159)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    required = (
        TRAIN_CHANNEL,
        MAP_FEATURES,
        FOLD_CACHE,
        OOF_META,
        POWER_CACHE,
        e153.V7_CHANNEL,
    )
    missing = [PATHS.relative(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing E159 inputs: {missing}")

    setup, positions, _, site, _ = load_positions()
    positions = positions.astype(np.float64)
    site = site.astype(np.int8)
    train_channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    point_power = np.load(POWER_CACHE).astype(np.float64)
    rms = e158.point_rms(point_power)
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    fit_ids = folds[f"fold{OUTER}_fit"].astype(np.int64)
    held_ids = folds[f"fold{OUTER}_query"].astype(np.int64)

    map_raw = np.load(MAP_FEATURES, allow_pickle=False)["train_features"].astype(
        np.float32
    )
    map_mean = np.mean(map_raw[fit_ids], axis=0, dtype=np.float64).astype(np.float32)
    map_std = np.maximum(
        np.std(map_raw[fit_ids], axis=0, dtype=np.float64).astype(np.float32),
        1e-4,
    )
    map_features = np.clip((map_raw - map_mean) / map_std, -6.0, 6.0).astype(
        np.float32
    )
    query_features = e158.build_query_features(
        positions, site, setup.bs_positions, map_features
    )
    site_scales = e158.build_site_scales(rms, site, fit_ids)

    train_table = e158.build_neighbor_table(
        fit_ids,
        fit_ids,
        positions,
        site,
        setup.bs_positions,
        map_features,
        query_features,
        rms,
        site_scales,
        config.neighbors,
        config.train_guard_m,
    )
    train_dataset = e158.WavefieldDataset(
        train_channel, train_table, site, site_scales, device
    )
    model, history = train_spectral_model(
        train_dataset, config, point_power, device
    )

    held_table = e158.build_neighbor_table(
        held_ids,
        fit_ids,
        positions,
        site,
        setup.bs_positions,
        map_features,
        query_features,
        rms,
        site_scales,
        config.neighbors,
        0.0,
    )
    held_dataset = e158.WavefieldDataset(
        train_channel, held_table, site, site_scales, device
    )
    spectral_prediction, spectral_metrics = e158.evaluate_model(
        model, held_dataset, True, batch_size=1
    )
    v7 = load_v7_channel(len(train_channel), held_ids)
    truth = np.asarray(train_channel[held_ids], dtype=np.complex64)
    v7_metrics = aggregate_statistics(point_metric_statistics(v7, truth))
    projected = project_spectra_onto_v7(v7, spectral_prediction, device)
    projected_metrics = aggregate_statistics(
        point_metric_statistics(projected, truth)
    )

    score_gain = float(projected_metrics["score"] - v7_metrics["score"])
    pas_gain = float(projected_metrics["pas"] - v7_metrics["pas"])
    pdp_gain = float(projected_metrics["pdp"] - v7_metrics["pdp"])
    nmse_delta = float(projected_metrics["nmse"] - v7_metrics["nmse"])
    gate_passed = bool(
        score_gain >= REQUIRED_SCORE_GAIN
        and pas_gain >= 0.0
        and pdp_gain >= 0.0
        and nmse_delta <= MAX_NMSE_DEGRADATION
    )

    spectral_path = ARTIFACT_DIR / "outer3_spectral_prediction.npy"
    projected_path = ARTIFACT_DIR / "outer3_v7_projected_prediction.npy"
    checkpoint_path = ARTIFACT_DIR / "outer3_spectral_operator.pt"
    np.save(spectral_path, spectral_prediction)
    np.save(projected_path, projected)
    save_checkpoint(checkpoint_path, model, config)

    payload = {
        "experiment": VERSION,
        "status": "screen_gate_passed" if gate_passed else "screen_gate_failed",
        "official_inputs_only": True,
        "sionna_used": False,
        "v12_used": False,
        "outer": OUTER,
        "held_count": int(len(held_ids)),
        "config": asdict(config),
        "projection": {
            "alpha_pas": ALPHA_PAS,
            "alpha_pdp": ALPHA_PDP,
            "rounds": PROJECTION_ROUNDS,
            "order": "pas_pdp",
            "preserve_v7_point_power": True,
        },
        "history": history,
        "metrics": {
            "v7": v7_metrics,
            "raw_spectral_prediction": spectral_metrics,
            "v7_projected": projected_metrics,
            "delta_vs_v7": {
                "pas": pas_gain,
                "pdp": pdp_gain,
                "nmse": nmse_delta,
                "score": score_gain,
            },
        },
        "gate": {
            "required_score_gain": REQUIRED_SCORE_GAIN,
            "require_pas_non_worse": True,
            "require_pdp_non_worse": True,
            "max_nmse_degradation": MAX_NMSE_DEGRADATION,
            "passed": gate_passed,
        },
        "outputs": {
            "spectral_prediction": PATHS.relative(spectral_path),
            "projected_prediction": PATHS.relative(projected_path),
            "checkpoint": PATHS.relative(checkpoint_path),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
