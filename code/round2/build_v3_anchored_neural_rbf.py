#!/usr/bin/env python3
"""Train the full V3 anchored neural RBF model and generate its test channel."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from data import load_positions
from pre_sionna_spectral import project_channel
from project_paths import PATHS
from run_rbf_spectral_field import ACTIONS, FOUNDATION_TEST
from run_v3_anchored_neural_rbf import (
    ARTIFACT_DIR,
    DESCRIPTORS,
    GEOMETRY,
    PAS_SIZE,
    AnchoredWeightCorrector,
    Config,
    build_train_contexts,
    forward_rows,
    power_cosine,
    set_seed,
    standardize_geometry,
    tensor_context,
)


ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v3_anchored_neural_rbf"
REPORT = ANALYSIS_DIR / "full_model_and_test.json"
SELECTION = ARTIFACT_DIR / "projection_selection.json"
V2_TEST_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "rbf_spectral_field"
    / "round2_rbf_test_weights.npz"
)
PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
TRAIN_AD = PRE_DIR / "round2_train_ad_support_f16.npy"
TRAIN_PAS = PRE_DIR / "round2_train_pas_link_support_f16.npy"
TRAIN_PDP = PRE_DIR / "round2_train_pdp_link_support_f16.npy"

FULL_CHECKPOINT = ARTIFACT_DIR / "full_epoch6.pt"
TEST_WEIGHTS = ARTIFACT_DIR / "test_corrected_weights.npz"
OUTPUT = ARTIFACT_DIR / "Round2_V3_Anchored_Neural_RBF_Test_Channel.npy"

PAS_GLOBAL_SHRINK = 0.10
PDP_GLOBAL_SHRINK = 0.55


def normalize(value: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=axis, keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, 1e-20),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def spectral_target(
    cache: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
    axes: int | tuple[int, ...],
) -> np.ndarray:
    donor = np.sqrt(np.maximum(np.asarray(cache[donor_ids], dtype=np.float32), 0.0))
    amplitude = np.maximum(
        np.tensordot(weights.astype(np.float32), donor, axes=(0, 0)), 0.0
    )
    return normalize(np.square(amplitude), axes)


def amplitude_blend(
    base: np.ndarray,
    shared: np.ndarray,
    alpha: float,
    axis: int,
) -> np.ndarray:
    amplitude = (1.0 - alpha) * np.sqrt(np.maximum(base, 0.0))
    amplitude += alpha * np.sqrt(np.maximum(shared, 0.0))
    return normalize(np.square(amplitude), axis)


def train_full_model(
    config: Config,
    device: torch.device,
) -> tuple[AnchoredWeightCorrector, dict[str, object]]:
    set_seed(config.seed + 90_000)
    _, train_positions, _, train_site, _ = load_positions()
    descriptors = np.load(DESCRIPTORS)
    valid = descriptors["valid"].astype(bool)
    spectra_numpy = np.concatenate(
        [
            np.sqrt(np.maximum(descriptors["global_pas"].astype(np.float32), 0.0)),
            np.sqrt(np.maximum(descriptors["global_pdp"].astype(np.float32), 0.0)),
        ],
        axis=1,
    ).astype(np.float32)
    geometry_numpy = np.load(GEOMETRY)["train"].astype(np.float32)
    fit_ids = np.arange(len(train_positions), dtype=np.int64)
    table = build_train_contexts(
        99, train_positions, train_site, valid, fit_ids, config
    )
    standardized, feature_mean, feature_std = standardize_geometry(
        geometry_numpy, fit_ids
    )
    geometry = torch.as_tensor(standardized, dtype=torch.float32, device=device)
    positions = torch.as_tensor(train_positions, dtype=torch.float32, device=device)
    spectra = torch.as_tensor(spectra_numpy, dtype=torch.float32, device=device)
    context = tensor_context(table, device)
    model = AnchoredWeightCorrector(geometry.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed + 90_000)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(table.query_ids))
        rows_epoch: list[tuple[float, float, float, float]] = []
        for start in range(0, len(order), config.batch_size):
            local = order[start : start + config.batch_size]
            rows = torch.as_tensor(local, dtype=torch.long, device=device)
            guard = torch.as_tensor(
                rng.integers(0, len(config.guard_radii_m), size=len(local)),
                dtype=torch.long,
                device=device,
            )
            pas, pdp, diagnostics = forward_rows(
                model, context, rows, guard, geometry, positions, spectra
            )
            query_ids = context["query_ids"][rows]
            pas_cosine = power_cosine(pas, spectra[query_ids, :PAS_SIZE])
            pdp_cosine = power_cosine(pdp, spectra[query_ids, PAS_SIZE:])
            correction = diagnostics["correction"]
            regularization = config.correction_regularization * (
                correction.square().sum(dim=1).mean()
                * config.neighbors
                / max(config.correction_l1_bound**2, 1e-8)
            )
            loss = 1.0 - 0.5 * (pas_cosine.mean() + pdp_cosine.mean())
            loss = loss + regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            rows_epoch.append(
                (
                    float(loss.detach().cpu()),
                    float(pas_cosine.mean().detach().cpu()),
                    float(pdp_cosine.mean().detach().cpu()),
                    float(correction.abs().sum(dim=1).mean().detach().cpu()),
                )
            )
        row = {
            "epoch": epoch,
            "loss": float(np.mean([value[0] for value in rows_epoch])),
            "pas": float(np.mean([value[1] for value in rows_epoch])),
            "pdp": float(np.mean([value[2] for value in rows_epoch])),
            "correction_l1": float(np.mean([value[3] for value in rows_epoch])),
        }
        history.append(row)
        print(f"V3 full epoch={epoch}/{config.epochs} {row}", flush=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "epoch": config.epochs,
            "seed": config.seed + 90_000,
        },
        FULL_CHECKPOINT,
    )
    return model, {
        "fit_count": int(len(fit_ids)),
        "train_valid_count": int(len(table.query_ids)),
        "history": history,
        "checkpoint": PATHS.relative(FULL_CHECKPOINT),
    }


@torch.no_grad()
def predict_test_weights(
    model: AnchoredWeightCorrector,
    checkpoint: dict[str, object],
    device: torch.device,
) -> dict[str, np.ndarray]:
    _, train_positions, test_positions, _, _ = load_positions()
    geometry_cache = np.load(GEOMETRY)
    selected_train = geometry_cache["train"].astype(np.float32)[:, 3:]
    selected_test = geometry_cache["test"].astype(np.float32)[:, 3:]
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    train_geometry = torch.as_tensor(
        (selected_train - feature_mean[None, :]) / feature_std[None, :],
        dtype=torch.float32,
        device=device,
    )
    test_geometry = torch.as_tensor(
        (selected_test - feature_mean[None, :]) / feature_std[None, :],
        dtype=torch.float32,
        device=device,
    )
    descriptors = np.load(DESCRIPTORS)
    spectra_numpy = np.concatenate(
        [
            np.sqrt(np.maximum(descriptors["global_pas"].astype(np.float32), 0.0)),
            np.sqrt(np.maximum(descriptors["global_pdp"].astype(np.float32), 0.0)),
        ],
        axis=1,
    ).astype(np.float32)
    spectra = torch.as_tensor(spectra_numpy, dtype=torch.float32, device=device)
    train_xy = torch.as_tensor(train_positions[:, :2], dtype=torch.float32, device=device)
    test_xy = torch.as_tensor(test_positions[:, :2], dtype=torch.float32, device=device)
    cached = np.load(V2_TEST_WEIGHTS)
    donor_ids = cached["donor_ids"].astype(np.int64)
    base_weights = cached["weights"].astype(np.float32)
    distances = np.linalg.norm(
        train_positions[donor_ids, :2] - test_positions[:, None, :2], axis=2
    ).astype(np.float32)
    pas_weights = np.empty_like(base_weights)
    pdp_weights = np.empty_like(base_weights)
    model.eval()
    for start in range(0, len(test_positions), 128):
        end = min(start + 128, len(test_positions))
        local_ids = torch.as_tensor(donor_ids[start:end], dtype=torch.long, device=device)
        local_distance = torch.as_tensor(
            distances[start:end], dtype=torch.float32, device=device
        )
        local_base = torch.as_tensor(
            base_weights[start:end], dtype=torch.float32, device=device
        )
        _, _, diagnostics = model(
            test_geometry[start:end],
            test_xy[start:end],
            train_geometry[local_ids],
            train_xy[local_ids],
            spectra[local_ids],
            local_distance,
            local_base,
        )
        pas_weights[start:end] = diagnostics["pas_weights"].cpu().numpy()
        pdp_weights[start:end] = diagnostics["pdp_weights"].cpu().numpy()
    np.savez_compressed(
        TEST_WEIGHTS,
        donor_ids=donor_ids.astype(np.int32),
        base_weights=base_weights,
        pas_weights=pas_weights,
        pdp_weights=pdp_weights,
        epoch=np.int32(6),
    )
    return {
        "donor_ids": donor_ids,
        "pas_weights": pas_weights,
        "pdp_weights": pdp_weights,
    }


def generate_test(
    weights: dict[str, np.ndarray],
    selection: dict[str, object],
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    donor_ids = weights["donor_ids"]
    pas_weights = weights["pas_weights"]
    pdp_weights = weights["pdp_weights"]
    train_ad = np.load(TRAIN_AD, mmap_mode="r")
    train_pas = np.load(TRAIN_PAS, mmap_mode="r")
    train_pdp = np.load(TRAIN_PDP, mmap_mode="r")
    base = np.load(FOUNDATION_TEST, mmap_mode="r")
    _, _, test_positions, _, test_site = load_positions()
    policy = {int(row["site"]): row for row in selection["final_policy"]}
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )

    for start in range(0, len(test_positions), batch_size):
        end = min(start + batch_size, len(test_positions))
        ad_rows: list[np.ndarray] = []
        pas_rows: list[np.ndarray] = []
        pdp_rows: list[np.ndarray] = []
        for row in range(start, end):
            ad = spectral_target(
                train_ad, donor_ids[row], pas_weights[row], (0, 1, 2)
            )
            pas = spectral_target(train_pas, donor_ids[row], pas_weights[row], 0)
            pdp = spectral_target(train_pdp, donor_ids[row], pdp_weights[row], 2)
            shared_pas = np.broadcast_to(
                pas.mean(axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32),
                pas.shape,
            )
            shared_pdp = np.broadcast_to(
                pdp.mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32),
                pdp.shape,
            )
            pas = amplitude_blend(pas, shared_pas, PAS_GLOBAL_SHRINK, 0)
            pdp = amplitude_blend(pdp, shared_pdp, PDP_GLOBAL_SHRINK, 2)
            ad_rows.append(ad)
            pas_rows.append(pas)
            pdp_rows.append(pdp)
        ad_tensor = torch.as_tensor(np.stack(ad_rows), device=device)
        pas_tensor = torch.as_tensor(np.stack(pas_rows), device=device)
        pdp_tensor = torch.as_tensor(np.stack(pdp_rows), device=device)
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        block = np.empty_like(np.asarray(base[start:end]))
        block_site = test_site[start:end]
        for site_value in np.unique(block_site):
            local = np.flatnonzero(block_site == site_value)
            row = policy[int(site_value)]
            action = ACTIONS[int(row["action_index"])]
            with torch.inference_mode():
                prediction = project_channel(
                    base_tensor[local],
                    ad_tensor[local],
                    pas_tensor[local],
                    pdp_tensor[local],
                    action.alpha_ad,
                    action.alpha_pas,
                    action.alpha_pdp,
                )
            block[local] = (
                prediction.cpu().numpy() * np.float32(row["scale"])
            ).astype(np.complex64)
        output[start:end] = block
        output.flush()
        print(f"V3 test {end}/{len(test_positions)}", flush=True)

    finite = True
    for start in range(0, len(output), 8):
        block = np.asarray(output[start : start + 8])
        finite = finite and bool(
            np.isfinite(block.real).all() and np.isfinite(block.imag).all()
        )
    if not finite:
        raise ValueError("V3 output contains non-finite values")
    return {
        "path": PATHS.relative(OUTPUT),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "bytes": int(OUTPUT.stat().st_size),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    config = Config(
        epochs=6,
        batch_size=128,
        correction_l1_bound=0.8,
        correction_regularization=0.003,
    )
    if args.skip_train:
        checkpoint = torch.load(FULL_CHECKPOINT, map_location=device, weights_only=False)
        model = AnchoredWeightCorrector(
            len(np.asarray(checkpoint["feature_mean"])), config
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        training = {
            "checkpoint": PATHS.relative(FULL_CHECKPOINT),
            "resumed": True,
        }
    else:
        model, training = train_full_model(config, device)
        checkpoint = torch.load(FULL_CHECKPOINT, map_location=device, weights_only=False)
    weights = predict_test_weights(model, checkpoint, device)
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    output = generate_test(weights, selection, device, args.batch_size)
    payload = {
        "experiment": "Round2-E4-V3-anchored-neural-RBF-full",
        "status": "validated_candidate_source",
        "primary_hypothesis": (
            "A bounded neural correction learns which official training spectra should "
            "compete inside the stable V2 RBF neighborhood."
        ),
        "sionna_used": False,
        "official_inputs_only": True,
        "config": asdict(config),
        "target_policy": selection["target_policy"],
        "local_validation": selection["nested_selected"],
        "local_fold_scores": selection["nested_fold_scores"],
        "final_policy": selection["final_policy"],
        "training": training,
        "output": output,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
