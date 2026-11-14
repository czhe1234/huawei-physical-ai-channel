#!/usr/bin/env python3
"""Learn a strictly cross-fitted low-rank PAS residual on the V10 carrier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_joint_path_token_oracle import OOF_META
from analyze_v11_explicit_path_pas_upper_bound import RAW_PAS, v10_oof_channel
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution, truth_views
from prepare_sionna_calibration_dataset import TRUTH_PAS
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics
from train_full_path_pas_slice_gate import project_joint


VERSION = "Round2-E74-low-rank-path-conditioned-PAS-residual"
PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n469_s24_k128_b096_d064_scatter1_sel106"
)
POINT_CONTEXT = (
    PATHS.artifacts / "round2" / "full_path_pdp_link_gate" / "train_point_context.npy"
)
PATH_CONTEXT = (
    PATHS.artifacts / "round2" / "path_augmented_pas_endpoint" / "train_path_context.npy"
)
E39_ROOT = PATHS.artifacts / "round2" / "sionna_multiseed"
E39_BASE = (
    E39_ROOT
    / "primary_d16_base_v4lambda_1m"
    / "full"
    / "checkpoints"
    / "n24_train_pas_mean.npy"
)
E39_SCATTER = (
    E39_ROOT
    / "primary_d10_scatter015_v4lambda_1m"
    / "full"
    / "checkpoints"
    / "n24_train_pas_mean.npy"
)
E40_PAS = (
    PATHS.artifacts / "round2" / "sionna_official_calibration" / "oof_pas_mean_f16.npy"
)
V9_PAS = (
    PATHS.artifacts / "round2" / "v9_joint_sionna_endpoints" / "oof_pas_joint_f16.npy"
)
ARTIFACT_DIR = PATHS.artifacts / "round2" / "low_rank_path_pas_residual"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_low_rank_path_pas_residual"
INCOHERENT_PAS = ARTIFACT_DIR / "oof_incoherent_path_pas_f16.npy"
REPORT = ANALYSIS_DIR / "low_rank_path_pas_residual.json"
TINY = 1e-30

SOURCE_NAMES = (
    "e40_calibrated",
    "v9_joint",
    "e39_base",
    "e39_scatter",
    "explicit_coherent",
    "explicit_incoherent",
)
ENDPOINT_SOURCES = (0, 1, 2, 3)
PATH_SOURCES = (4, 5)
GROUPS = ("control", "endpoint", "path", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--frequency-stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ridge", default="0.003,0.03,0.3")
    parser.add_argument("--eta", default="0,0.10,0.25,0.50,0.75,1.00")
    parser.add_argument("--force-incoherent", action="store_true")
    return parser.parse_args()


def parse_grid(value: str) -> tuple[float, ...]:
    output = tuple(float(item) for item in value.split(",") if item.strip())
    if not output:
        raise ValueError("empty numeric grid")
    return output


def normalize_pas(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=(1, 2, 3), keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), 0.0)


def pas_slices(value: torch.Tensor) -> torch.Tensor:
    return value.permute(0, 4, 5, 1, 2, 3).reshape(len(value), 4, 192, 256)


def pas_from_slices(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(len(value), 4, 192, 2, 16, 8).permute(0, 3, 4, 5, 1, 2)


def slice_cosine(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=3, dtype=torch.float64)
    denominator = torch.sqrt(
        prediction.square().sum(dim=3, dtype=torch.float64)
        * truth.square().sum(dim=3, dtype=torch.float64)
    )
    return numerator / torch.where(denominator == 0.0, 1e-12, denominator)


def build_incoherent_pas(args: argparse.Namespace, device: torch.device) -> np.ndarray:
    if INCOHERENT_PAS.is_file() and not args.force_incoherent:
        return np.load(INCOHERENT_PAS, mmap_mode="r")
    coefficient = np.load(PATH_CACHE / "coefficient.npy", mmap_mode="r")
    mask = np.load(PATH_CACHE / "mask.npy", mmap_mode="r")
    source = np.load(PATH_CACHE / "source.npy", mmap_mode="r")
    fraction = np.load(PATH_CACHE / "energy_fraction.npy", mmap_mode="r")
    output = np.lib.format.open_memmap(
        INCOHERENT_PAS,
        mode="w+",
        dtype=np.float16,
        shape=(len(coefficient), 2, 16, 8, 4),
    )
    for start in range(0, len(coefficient), args.batch_size):
        end = min(start + args.batch_size, len(coefficient))
        local = torch.as_tensor(
            np.array(coefficient[start:end], dtype=np.complex64, copy=True),
            device=device,
        ).reshape(end - start, 128, 2, 16, 8, 4)
        power = torch.fft.fft2(local, dim=(3, 4)).abs().square().float()
        power /= power.sum(dim=(2, 3, 4), keepdim=True).clamp_min(TINY)
        local_mask = torch.as_tensor(np.asarray(mask[start:end]), device=device)
        local_source = torch.as_tensor(np.asarray(source[start:end]), device=device)
        local_fraction = torch.as_tensor(
            np.asarray(fraction[start:end], dtype=np.float32), device=device
        )
        # Base paths are de-duplicated across seeds. Scatter paths are samples from
        # 24 independent realizations, so average their power instead of coherently
        # summing their complex coefficients.
        source_weight = torch.where(
            local_source == 0,
            torch.ones_like(local_fraction),
            torch.where(
                local_source == 1,
                torch.full_like(local_fraction, 1.0 / 24.0),
                torch.zeros_like(local_fraction),
            ),
        )
        weight = local_mask * local_fraction * source_weight
        result = (power * weight[..., None, None, None, None]).sum(dim=1)
        result /= result.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)
        output[start:end] = result.cpu().numpy().astype(np.float16)
        if end % 100 == 0 or end == len(coefficient):
            output.flush()
            print(f"incoherent path PAS {end}/{len(coefficient)}", flush=True)
    output.flush()
    return np.load(INCOHERENT_PAS, mmap_mode="r")


def load_sources(
    rows: np.ndarray,
    global_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> list[torch.Tensor]:
    arrays: list[np.ndarray] = [
        np.asarray(np.load(E40_PAS, mmap_mode="r")[rows], dtype=np.float32),
        np.asarray(np.load(V9_PAS, mmap_mode="r")[rows], dtype=np.float32),
        np.asarray(np.load(E39_BASE, mmap_mode="r")[global_ids], dtype=np.float32),
        np.asarray(np.load(E39_SCATTER, mmap_mode="r")[global_ids], dtype=np.float32),
        np.asarray(np.load(RAW_PAS, mmap_mode="r"), dtype=np.float32),
    ]
    # E38 established a fixed +2 vertical-bin scorer alignment for 4-lambda PAS.
    arrays[2] = np.roll(arrays[2], 2, axis=3)
    arrays[3] = np.roll(arrays[3], 2, axis=3)
    incoherent = np.asarray(build_incoherent_pas(args, device), dtype=np.float32)
    arrays.append(np.broadcast_to(incoherent[..., None], incoherent.shape + (192,)))
    output = []
    for value in arrays:
        tensor = normalize_pas(torch.as_tensor(np.array(value, copy=True), device=device))
        output.append(torch.sqrt(pas_slices(tensor).clamp_min(0.0)))
    return output


def coordinate_features(device: torch.device) -> torch.Tensor:
    ue = torch.arange(4, device=device, dtype=torch.long)
    frequency = torch.arange(192, device=device, dtype=torch.float32)
    phase = 2.0 * torch.pi * frequency / 192.0
    ue_block = torch.eye(4, device=device)[ue][:, None].expand(4, 192, 4)
    frequency_block = torch.stack(
        (
            (frequency - 95.5) / 95.5,
            torch.sin(phase),
            torch.cos(phase),
            torch.sin(2.0 * phase),
            torch.cos(2.0 * phase),
        ),
        dim=1,
    )[None].expand(4, -1, -1)
    return torch.cat((ue_block, frequency_block), dim=2)


def fit_basis(
    residual: torch.Tensor,
    fit_points: np.ndarray,
    rank: int,
    frequency_stride: int,
) -> tuple[torch.Tensor, np.ndarray]:
    local = residual[
        torch.as_tensor(fit_points, device=residual.device),
        :,
        ::frequency_stride,
    ].reshape(-1, 256)
    covariance = local.T @ local / max(len(local), 1)
    values, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True)[:rank]
    basis = vectors[:, order].T.contiguous()
    selected = values[order].clamp_min(0.0)
    ratio = (selected / values.clamp_min(0.0).sum().clamp_min(TINY)).cpu().numpy()
    return basis, ratio


def repeated_context(context: torch.Tensor) -> torch.Tensor:
    return context[:, None, None].expand(-1, 4, 192, -1)


def build_feature_blocks(
    base_amplitude: torch.Tensor,
    sources: list[torch.Tensor],
    basis: torch.Tensor,
    point_context: torch.Tensor,
    path_context: torch.Tensor,
) -> dict[str, torch.Tensor]:
    base_power = base_amplitude.square()
    base_mode = torch.einsum("pusd,kd->pusk", base_amplitude, basis)
    base_entropy = -(
        base_power * torch.log(base_power.clamp_min(1e-12))
    ).sum(dim=3, keepdim=True) / np.log(256.0)
    base_peak = base_power.max(dim=3, keepdim=True).values
    coordinates = coordinate_features(base_amplitude.device)[None].expand(
        len(base_amplitude), -1, -1, -1
    )
    geometry = repeated_context(point_context[:, 120:])
    common = torch.cat((base_mode, base_entropy, base_peak, coordinates, geometry), dim=3)

    source_blocks = []
    for source in sources:
        delta = source - base_amplitude
        coefficient = torch.einsum("pusd,kd->pusk", delta, basis)
        source_power = source.square()
        agreement = (source * base_amplitude).sum(dim=3, keepdim=True)
        norm = delta.square().sum(dim=3, keepdim=True).sqrt()
        peak = source_power.max(dim=3, keepdim=True).values
        entropy = -(
            source_power * torch.log(source_power.clamp_min(1e-12))
        ).sum(dim=3, keepdim=True) / np.log(256.0)
        valid = (source_power.sum(dim=3, keepdim=True) > TINY).float()
        source_blocks.append(
            torch.cat((coefficient, agreement, norm, peak, entropy, valid), dim=3)
        )
    endpoint = torch.cat([source_blocks[index] for index in ENDPOINT_SOURCES], dim=3)
    path = torch.cat([source_blocks[index] for index in PATH_SOURCES], dim=3)
    path_summary = repeated_context(torch.cat((point_context[:, :120], path_context), dim=1))
    return {
        "control": common,
        "endpoint": torch.cat((common, endpoint), dim=3),
        "path": torch.cat((common, path, path_summary), dim=3),
        "all": torch.cat((common, endpoint, path, path_summary), dim=3),
    }


def sample_rows(
    value: torch.Tensor,
    point_mask: np.ndarray,
    frequency_stride: int,
) -> torch.Tensor:
    ids = torch.as_tensor(np.flatnonzero(point_mask), device=value.device)
    return value[ids, :, ::frequency_stride].reshape(-1, value.shape[-1])


def predict_coefficients(
    x: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return ((x - x_mean) / x_std) @ weight + y_mean


def target_from_coefficients(
    base: torch.Tensor,
    coefficients: torch.Tensor,
    basis: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    residual = torch.einsum("pusk,kd->pusd", coefficients, basis)
    amplitude = (base + eta * residual).clamp_min(0.0)
    power = amplitude.square()
    return power / power.sum(dim=3, keepdim=True).clamp_min(TINY)


def endpoint_summary(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    fold_id: np.ndarray,
    baseline: torch.Tensor,
) -> dict[str, object]:
    values = slice_cosine(prediction, truth).mean(dim=(1, 2)).cpu().numpy()
    anchors = slice_cosine(baseline, truth).mean(dim=(1, 2)).cpu().numpy()
    folds = []
    for fold in range(5):
        mask = fold_id == fold
        folds.append(
            {
                "fold": fold,
                "baseline": float(np.mean(anchors[mask])),
                "selected": float(np.mean(values[mask])),
                "delta": float(np.mean(values[mask]) - np.mean(anchors[mask])),
            }
        )
    return {
        "baseline": float(np.mean(anchors)),
        "selected": float(np.mean(values)),
        "delta": float(np.mean(values) - np.mean(anchors)),
        "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
        "folds": folds,
    }


def choose_nested_action(
    group: str,
    outer: int,
    features: torch.Tensor,
    target_coefficients: torch.Tensor,
    base: torch.Tensor,
    truth: torch.Tensor,
    basis: torch.Tensor,
    fold_id: np.ndarray,
    ridge_grid: tuple[float, ...],
    eta_grid: tuple[float, ...],
    frequency_stride: int,
) -> tuple[float, float, list[dict[str, object]]]:
    candidates = {
        (penalty, eta): [] for penalty in ridge_grid for eta in eta_grid
    }
    for inner in range(5):
        if inner == outer:
            continue
        train_mask = (fold_id != outer) & (fold_id != inner)
        validation_mask = fold_id == inner
        x_train = sample_rows(features, train_mask, frequency_stride)
        y_train = sample_rows(target_coefficients, train_mask, frequency_stride)
        x_validation = features[
            torch.as_tensor(np.flatnonzero(validation_mask), device=features.device)
        ]
        local_base = base[
            torch.as_tensor(np.flatnonzero(validation_mask), device=features.device)
        ]
        local_truth = truth[
            torch.as_tensor(np.flatnonzero(validation_mask), device=features.device)
        ]
        anchor = float(slice_cosine(local_base.square(), local_truth.square()).mean().item())
        x_mean = x_train.mean(dim=0)
        x_std = x_train.std(dim=0, unbiased=False)
        x_std = torch.where(x_std > 1e-5, x_std, torch.ones_like(x_std))
        y_mean = y_train.mean(dim=0)
        x_standardized = (x_train - x_mean) / x_std
        covariance = x_standardized.T @ x_standardized / max(len(x_train), 1)
        cross = x_standardized.T @ (y_train - y_mean) / max(len(x_train), 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        projected_cross = eigenvectors.T @ cross
        for penalty in ridge_grid:
            weight = eigenvectors @ (
                projected_cross / (eigenvalues[:, None] + penalty)
            )
            coefficients = predict_coefficients(
                x_validation, x_mean, x_std, y_mean, weight
            )
            for eta in eta_grid:
                prediction = target_from_coefficients(
                    local_base, coefficients, basis, eta
                )
                selected = float(
                    slice_cosine(prediction, local_truth.square()).mean().item()
                )
                candidates[(penalty, eta)].append(
                    {
                        "inner_fold": inner,
                        "baseline": anchor,
                        "selected": selected,
                        "delta": selected - anchor,
                    }
                )
        del x_train, y_train, x_validation, x_standardized, covariance, cross
    rows = []
    for (penalty, eta), folds in candidates.items():
        weighted = sum(row["selected"] for row in folds) / max(len(folds), 1)
        baseline = sum(row["baseline"] for row in folds) / max(len(folds), 1)
        rows.append(
            {
                "group": group,
                "ridge": penalty,
                "eta": eta,
                "baseline": baseline,
                "selected": weighted,
                "delta": weighted - baseline,
                "positive_inner_folds": int(sum(row["delta"] > 0.0 for row in folds)),
                "inner_folds": folds,
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["positive_inner_folds"] >= 3),
            float(row["selected"]),
            -float(row["eta"]),
        ),
        reverse=True,
    )
    winner = rows[0]
    print(
        f"outer {outer} {group}: ridge={winner['ridge']} eta={winner['eta']} "
        f"inner_delta={winner['delta']:+.6f} "
        f"positive={winner['positive_inner_folds']}/4",
        flush=True,
    )
    return float(winner["ridge"]), float(winner["eta"]), rows


def fit_final_model(
    features: torch.Tensor,
    target_coefficients: torch.Tensor,
    fit_mask: np.ndarray,
    held_mask: np.ndarray,
    penalty: float,
    frequency_stride: int,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    x_train = sample_rows(features, fit_mask, frequency_stride)
    y_train = sample_rows(target_coefficients, fit_mask, frequency_stride)
    x_mean = x_train.mean(dim=0)
    x_std = x_train.std(dim=0, unbiased=False)
    x_std = torch.where(x_std > 1e-5, x_std, torch.ones_like(x_std))
    y_mean = y_train.mean(dim=0)
    standardized = (x_train - x_mean) / x_std
    covariance = standardized.T @ standardized / max(len(x_train), 1)
    cross = standardized.T @ (y_train - y_mean) / max(len(x_train), 1)
    weight = torch.linalg.solve(
        covariance + penalty * torch.eye(covariance.shape[0], device=covariance.device),
        cross,
    )
    held = features[torch.as_tensor(np.flatnonzero(held_mask), device=features.device)]
    prediction = predict_coefficients(held, x_mean, x_std, y_mean, weight)
    state = {
        "x_mean": x_mean.cpu().numpy().astype(np.float32),
        "x_std": x_std.cpu().numpy().astype(np.float32),
        "y_mean": y_mean.cpu().numpy().astype(np.float32),
        "weight": weight.cpu().numpy().astype(np.float32),
    }
    return prediction, state


def full_channel_summary(
    base_channel: torch.Tensor,
    desired_pas: torch.Tensor,
    truth_channel: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
    fold_id: np.ndarray,
) -> dict[str, object]:
    base_pdp = pdp_distribution(base_channel)
    with torch.inference_mode():
        prediction = project_joint(
            base_channel, desired_pas, base_pdp, cycles=2, order="pas_pdp"
        )
        baseline_values = point_metrics(
            base_channel, truth_channel, truth_pas, truth_pdp
        )
        selected_values = point_metrics(
            prediction, truth_channel, truth_pas, truth_pdp
        )
    baseline = statistics(*baseline_values)
    selected = statistics(*selected_values)
    folds = []
    for fold in range(5):
        mask = fold_id == fold
        anchor = statistics(*baseline_values, mask)
        result = statistics(*selected_values, mask)
        folds.append(
            {
                "fold": fold,
                "baseline": anchor,
                "selected": result,
                "delta": float(result["score"] - anchor["score"]),
            }
        )
    return {
        "baseline": baseline,
        "selected": selected,
        "delta": float(selected["score"] - baseline["score"]),
        "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
        "folds": folds,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    ridge_grid = parse_grid(args.ridge)
    eta_grid = parse_grid(args.eta)
    if args.rank <= 0 or args.rank > 64:
        raise ValueError("rank must be in [1, 64]")
    if args.frequency_stride <= 0:
        raise ValueError("frequency stride must be positive")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (
        OOF_META,
        PATH_CACHE,
        POINT_CONTEXT,
        PATH_CONTEXT,
        E39_BASE,
        E39_SCATTER,
        E40_PAS,
        V9_PAS,
        RAW_PAS,
        TRUTH_PAS,
    )
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    metadata = np.load(OOF_META, allow_pickle=False)
    rows = np.load(PATH_CACHE / "rows.npy").astype(np.int64)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)[rows]
    global_ids = query_ids[rows]

    base_channel_array = v10_oof_channel(device)
    base_channel = torch.as_tensor(base_channel_array, device=device)
    base_pas = normalize_pas(pas_distribution(base_channel))
    base_amplitude = torch.sqrt(pas_slices(base_pas).clamp_min(0.0))
    truth_pas_array = np.load(TRUTH_PAS, mmap_mode="r")
    truth_pas = normalize_pas(
        torch.as_tensor(
            np.asarray(truth_pas_array[global_ids], dtype=np.float32), device=device
        )
    )
    truth_amplitude = torch.sqrt(pas_slices(truth_pas).clamp_min(0.0))
    residual = truth_amplitude - base_amplitude
    sources = load_sources(rows, global_ids, args, device)
    full_context = np.load(POINT_CONTEXT, mmap_mode="r")
    point_context = torch.as_tensor(
        np.asarray(full_context[global_ids], dtype=np.float32), device=device
    )
    full_path_context = np.load(PATH_CONTEXT, mmap_mode="r")
    path_context = torch.as_tensor(
        np.asarray(full_path_context[global_ids], dtype=np.float32), device=device
    )

    residual_outputs = {
        group: np.lib.format.open_memmap(
            ARTIFACT_DIR / f"oof_{group}_sqrt_residual_f16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(rows), 4, 192, 256),
        )
        for group in GROUPS
    }
    decisions: dict[str, list[dict[str, object]]] = {group: [] for group in GROUPS}
    basis_rows = []
    for outer in range(5):
        fit_mask = fold_id != outer
        held_mask = fold_id == outer
        basis, explained = fit_basis(
            residual,
            np.flatnonzero(fit_mask),
            args.rank,
            args.frequency_stride,
        )
        target_coefficients = torch.einsum("pusd,kd->pusk", residual, basis)
        blocks = build_feature_blocks(
            base_amplitude, sources, basis, point_context, path_context
        )
        basis_rows.append(
            {
                "fold": outer,
                "explained_sum": float(np.sum(explained)),
                "explained": explained.tolist(),
            }
        )
        held_ids = np.flatnonzero(held_mask)
        for group in GROUPS:
            penalty, eta, nested_rows = choose_nested_action(
                group,
                outer,
                blocks[group],
                target_coefficients,
                base_amplitude,
                truth_amplitude,
                basis,
                fold_id,
                ridge_grid,
                eta_grid,
                args.frequency_stride,
            )
            coefficients, state = fit_final_model(
                blocks[group],
                target_coefficients,
                fit_mask,
                held_mask,
                penalty,
                args.frequency_stride,
            )
            local_residual = eta * torch.einsum("pusk,kd->pusd", coefficients, basis)
            residual_outputs[group][held_ids] = (
                local_residual.cpu().numpy().astype(np.float16)
            )
            model_path = ARTIFACT_DIR / f"fold{outer}_{group}_ridge_model.npz"
            np.savez_compressed(
                model_path,
                basis=basis.cpu().numpy().astype(np.float32),
                penalty=np.asarray(penalty, dtype=np.float64),
                eta=np.asarray(eta, dtype=np.float64),
                **state,
            )
            decisions[group].append(
                {
                    "fold": outer,
                    "ridge": penalty,
                    "eta": eta,
                    "feature_dim": int(blocks[group].shape[-1]),
                    "fit_points": int(np.sum(fit_mask)),
                    "held_points": int(np.sum(held_mask)),
                    "model": PATHS.relative(model_path),
                    "nested_candidates": nested_rows,
                }
            )
        for output in residual_outputs.values():
            output.flush()
        del basis, target_coefficients, blocks
        if device.type == "cuda":
            torch.cuda.empty_cache()

    truth_channel_all = np.load(
        PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r"
    )
    truth_channel = torch.as_tensor(
        np.asarray(truth_channel_all[global_ids], dtype=np.complex64), device=device
    )
    truth_pas_view, truth_pdp_view = truth_views(truth_channel)
    group_results = {}
    for group in GROUPS:
        local_residual = torch.as_tensor(
            np.asarray(residual_outputs[group], dtype=np.float32), device=device
        )
        desired_slices = (base_amplitude + local_residual).clamp_min(0.0).square()
        desired_slices /= desired_slices.sum(dim=3, keepdim=True).clamp_min(TINY)
        desired_pas = pas_from_slices(desired_slices)
        group_results[group] = {
            "endpoint": endpoint_summary(
                desired_slices, truth_amplitude.square(), fold_id, base_amplitude.square()
            ),
            "full_channel": full_channel_summary(
                base_channel,
                desired_pas,
                truth_channel,
                truth_pas_view,
                truth_pdp_view,
                fold_id,
            ),
        }
        print(
            f"{group}: endpoint_delta={group_results[group]['endpoint']['delta']:+.6f} "
            f"full_delta={group_results[group]['full_channel']['delta']:+.6f} "
            f"positive={group_results[group]['full_channel']['positive_folds']}/5",
            flush=True,
        )

    all_delta = float(group_results["all"]["full_channel"]["delta"])
    control_delta = float(group_results["control"]["full_channel"]["delta"])
    path_delta = float(group_results["path"]["full_channel"]["delta"])
    passed = bool(
        group_results["all"]["full_channel"]["positive_folds"] >= 4
        and all_delta >= 5e-4
        and all_delta > control_delta + 1e-4
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Explicit propagation paths can explain a small number of V10 PAS residual "
            "modes even when their raw PAS endpoint is globally misaligned."
        ),
        "configuration": {
            **vars(args),
            "ridge_grid": ridge_grid,
            "eta_grid": eta_grid,
            "source_names": SOURCE_NAMES,
            "groups": GROUPS,
            "projection": "PAS->PDP, 2 cycles, V10 PDP and point power fixed",
        },
        "point_count": int(len(rows)),
        "basis": basis_rows,
        "decisions": decisions,
        "results": group_results,
        "path_information_margin": {
            "path_minus_control": path_delta - control_delta,
            "all_minus_control": all_delta - control_delta,
        },
        "deployment_gate": {
            "minimum_full_score_delta": 5e-4,
            "minimum_positive_folds": 4,
            "minimum_all_minus_control": 1e-4,
            "passed": passed,
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["deployment_gate"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
