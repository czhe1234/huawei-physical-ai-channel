"""Rebuild grouped calibration and exact-projection policy assets from OOF data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import torch

from .base_runtime.calibrated_exact_projection import mode_targets
from .base_runtime.donor_encoder import spectral_target
from .base_runtime.exact_projection import (
    ACTIONS,
    continue_trajectory,
    initialize_trajectory,
)
from .base_runtime.grouped_prediction import (
    blend_group_pas,
    blend_group_pdp,
    encode,
    globally_shrunk_targets,
    normalize_group_amplitude,
    predict_grouped,
)
from .carrier_training import oof_foundation_path, oof_mixture_path
from .config import PipelineConfig, RuntimePaths
from .data import load_official_data
from .donor_training import oof_weights_path
from .io_utils import atomic_json, atomic_npz, portable_path
from .training import prepare_training_inputs
from .training_common import FoldMetadata, load_fold_metadata


TINY = 1e-30
TARGET_MODES = ("raw", "pascal", "pdpcal", "bothcal")
ORDERS = ("pas_pdp", "pdp_pas")
CHECKPOINTS = (1, 2, 4, 8, 16)
EXACT_PATTERN = re.compile(
    r"^exact_(raw|pascal|pdpcal|bothcal)_(pas_pdp|pdp_pas)_(.+)_c(\d+)$"
)


@dataclass(frozen=True)
class SpectralAssetConfig:
    pas_components: int = 128
    pdp_components: int = 64
    pas_calibration_strength: float = 1.0
    pdp_calibration_strength: float = 0.5
    pas_group_blend: float = 0.25
    pdp_group_blend: float = 1.0
    metric_batch_size: int = 2
    seed: int = 20_260_805


def _root(paths: RuntimePaths) -> Path:
    return paths.work_dir / "training" / "spectral_assets"


def oof_grouped_pas_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "oof_calibrated_grouped_pas.npy"


def oof_grouped_pdp_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "oof_calibrated_grouped_pdp.npy"


def group_calibration_path(paths: RuntimePaths) -> Path:
    return paths.model_dir / "base_field" / "group_calibration.npz"


def _target_path(paths: RuntimePaths, domain: str) -> Path:
    return _root(paths) / "targets" / f"oof_{domain}.npy"


def projection_metrics_path(paths: RuntimePaths, name: str) -> Path:
    return _root(paths) / "projection" / f"{name}_metrics.npz"


def oof_reference_projection_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "projection" / "oof_reference_projection.npy"


def oof_final_projection_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "projection" / "oof_final_projection.npy"


def projection_policy_path(paths: RuntimePaths) -> Path:
    return paths.model_dir / "base_field" / "projection_policy.json"


def _pca_basis(
    fit_amplitude: np.ndarray,
    components: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    value = torch.as_tensor(fit_amplitude, dtype=torch.float32, device=device)
    mean = value.mean(dim=0)
    centered = value - mean
    count = min(components, len(value) - 1, value.shape[1])
    if count < 1:
        raise RuntimeError("PCA calibration requires at least two fitting rows")
    _, _, vectors = torch.pca_lowrank(centered, q=count, center=False, niter=4)
    return mean.cpu().numpy(), vectors.cpu().numpy()


def _fit_slope(
    amplitude_flat: np.ndarray,
    prediction: np.ndarray,
    query_ids: np.ndarray,
    calibration_rows: np.ndarray,
    basis_fit_ids: np.ndarray,
    components: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, vectors = _pca_basis(
        amplitude_flat[basis_fit_ids], components, device, seed
    )
    predicted = encode(
        prediction[calibration_rows].reshape(len(calibration_rows), -1),
        mean,
        vectors,
    )
    truth = encode(
        amplitude_flat[query_ids[calibration_rows]], mean, vectors
    )
    denominator = np.sum(predicted * predicted, axis=0, dtype=np.float64)
    slope = np.divide(
        np.sum(predicted * truth, axis=0, dtype=np.float64),
        np.maximum(denominator, 1e-12),
    )
    return (
        mean.astype(np.float32),
        vectors.astype(np.float32),
        np.clip(slope, 0.0, 1.25).astype(np.float32),
    )


def _apply_calibration(
    prediction: np.ndarray,
    rows: np.ndarray,
    model: tuple[np.ndarray, np.ndarray, np.ndarray],
    strength: float,
) -> np.ndarray:
    mean, vectors, slope = model
    flat = prediction[rows].reshape(len(rows), -1)
    coefficients = encode(flat, mean, vectors)
    gain = 1.0 + strength * (slope - 1.0)
    corrected = flat + (coefficients * gain - coefficients) @ vectors.T
    return normalize_group_amplitude(
        corrected.reshape(len(rows), *prediction.shape[1:])
    )


def _atomic_memmap(
    path: Path, dtype: np.dtype[Any], shape: tuple[int, ...]
) -> tuple[np.memmap, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    return np.lib.format.open_memmap(partial, mode="w+", dtype=dtype, shape=shape), partial


def build_group_calibration(
    paths: RuntimePaths,
    folds: FoldMetadata,
    device: torch.device,
    config: SpectralAssetConfig,
    force: bool,
) -> dict[str, Any]:
    pas_output = oof_grouped_pas_path(paths)
    pdp_output = oof_grouped_pdp_path(paths)
    model_output = group_calibration_path(paths)
    if pas_output.is_file() and pdp_output.is_file() and model_output.is_file() and not force:
        return {
            "status": "reused",
            "oof_pas": portable_path(pas_output),
            "oof_pdp": portable_path(pdp_output),
            "model": portable_path(model_output),
        }

    data = load_official_data(paths.data_dir)
    with np.load(oof_weights_path(paths), allow_pickle=False) as weights:
        query_ids = weights["query_ids"].astype(np.int64)
        donor_ids = weights["donor_ids"].astype(np.int64)
        pas_weights = weights["pas_weights"].astype(np.float32)
        pdp_weights = weights["pdp_weights"].astype(np.float32)
    if not np.array_equal(query_ids, folds.query_ids):
        raise RuntimeError("donor and spatial-fold OOF query order differs")
    grouped_pas = np.load(
        paths.base_dir / "official_spectra" / "grouped_pas.npy", mmap_mode="r"
    )
    grouped_pdp = np.load(
        paths.base_dir / "official_spectra" / "grouped_pdp.npy", mmap_mode="r"
    )
    prediction_pas = predict_grouped(grouped_pas, donor_ids, pas_weights)
    prediction_pdp = predict_grouped(grouped_pdp, donor_ids, pdp_weights)
    pas_flat = np.asarray(grouped_pas, dtype=np.float32).reshape(len(grouped_pas), -1)
    pdp_flat = np.asarray(grouped_pdp, dtype=np.float32).reshape(len(grouped_pdp), -1)
    with np.load(
        paths.base_dir / "official_spectra" / "descriptors.npz", allow_pickle=False
    ) as descriptors:
        valid = descriptors["valid"].astype(bool)
    query_site = data.train_sites[query_ids]
    oof_pas = np.empty_like(prediction_pas, dtype=np.float32)
    oof_pdp = np.empty_like(prediction_pdp, dtype=np.float32)
    records: list[dict[str, Any]] = []

    for fold in range(5):
        held_rows = np.flatnonzero(folds.fold_id == fold)
        fit_ids = folds.fit_indices[fold]
        fit_membership = np.zeros(len(data.train_positions), dtype=bool)
        fit_membership[fit_ids] = True
        for site in (0, 1):
            local_held = held_rows[query_site[held_rows] == site]
            if not len(local_held):
                continue
            calibration_rows = np.flatnonzero(
                (folds.fold_id != fold)
                & (query_site == site)
                & fit_membership[query_ids]
            )
            basis_fit = fit_ids[(data.train_sites[fit_ids] == site) & valid[fit_ids]]
            if len(calibration_rows) < 8:
                oof_pas[local_held] = prediction_pas[local_held]
                oof_pdp[local_held] = prediction_pdp[local_held]
                records.append(
                    {
                        "fold": fold,
                        "site": site,
                        "status": "identity_insufficient_calibration",
                        "calibration_count": int(len(calibration_rows)),
                    }
                )
                continue
            pas_model = _fit_slope(
                pas_flat,
                prediction_pas,
                query_ids,
                calibration_rows,
                basis_fit,
                config.pas_components,
                device,
                config.seed + fold * 10_000 + site * 1009,
            )
            pdp_model = _fit_slope(
                pdp_flat,
                prediction_pdp,
                query_ids,
                calibration_rows,
                basis_fit,
                config.pdp_components,
                device,
                config.seed + 1_000_000 + fold * 10_000 + site * 1009,
            )
            oof_pas[local_held] = _apply_calibration(
                prediction_pas,
                local_held,
                pas_model,
                config.pas_calibration_strength,
            )
            oof_pdp[local_held] = _apply_calibration(
                prediction_pdp,
                local_held,
                pdp_model,
                config.pdp_calibration_strength,
            )
            records.append(
                {
                    "fold": fold,
                    "site": site,
                    "status": "calibrated",
                    "basis_count": int(len(basis_fit)),
                    "calibration_count": int(len(calibration_rows)),
                    "held_count": int(len(local_held)),
                    "pas_slope_mean": float(np.mean(pas_model[2])),
                    "pdp_slope_mean": float(np.mean(pdp_model[2])),
                }
            )

    pas_memmap, pas_partial = _atomic_memmap(
        pas_output, np.dtype(np.float16), oof_pas.shape
    )
    pas_memmap[:] = oof_pas.astype(np.float16)
    pas_memmap.flush()
    del pas_memmap
    os.replace(pas_partial, pas_output)
    pdp_memmap, pdp_partial = _atomic_memmap(
        pdp_output, np.dtype(np.float16), oof_pdp.shape
    )
    pdp_memmap[:] = oof_pdp.astype(np.float16)
    pdp_memmap.flush()
    del pdp_memmap
    os.replace(pdp_partial, pdp_output)

    model_arrays: dict[str, np.ndarray] = {}
    production_records = []
    all_rows = np.arange(len(query_ids), dtype=np.int64)
    for site in (0, 1):
        basis_fit = np.flatnonzero((data.train_sites == site) & valid)
        calibration_rows = all_rows[query_site == site]
        pas_model = _fit_slope(
            pas_flat,
            prediction_pas,
            query_ids,
            calibration_rows,
            basis_fit,
            config.pas_components,
            device,
            config.seed + site * 1009,
        )
        pdp_model = _fit_slope(
            pdp_flat,
            prediction_pdp,
            query_ids,
            calibration_rows,
            basis_fit,
            config.pdp_components,
            device,
            config.seed + 1_000_000 + site * 1009,
        )
        prefix = f"site{site}"
        for domain, model in (("pas", pas_model), ("pdp", pdp_model)):
            model_arrays[f"{prefix}_{domain}_mean"] = model[0]
            model_arrays[f"{prefix}_{domain}_vectors"] = model[1]
            model_arrays[f"{prefix}_{domain}_slope"] = model[2]
        production_records.append(
            {
                "site": site,
                "basis_count": int(len(basis_fit)),
                "calibration_count": int(len(calibration_rows)),
                "pas_slope_mean": float(np.mean(pas_model[2])),
                "pdp_slope_mean": float(np.mean(pdp_model[2])),
            }
        )
    atomic_npz(model_output, **model_arrays)
    return {
        "status": "complete",
        "oof_pas": portable_path(pas_output),
        "oof_pdp": portable_path(pdp_output),
        "model": portable_path(model_output),
        "folds": records,
        "production": production_records,
    }


def build_oof_spectral_targets(
    paths: RuntimePaths, folds: FoldMetadata, force: bool
) -> dict[str, Path]:
    outputs = {name: _target_path(paths, name) for name in ("ad", "pas", "pdp")}
    if all(path.is_file() for path in outputs.values()) and not force:
        return outputs
    with np.load(oof_weights_path(paths), allow_pickle=False) as weights:
        query_ids = weights["query_ids"].astype(np.int64)
        donor_ids = weights["donor_ids"].astype(np.int64)
        pas_weights = weights["pas_weights"].astype(np.float32)
        pdp_weights = weights["pdp_weights"].astype(np.float32)
    if not np.array_equal(query_ids, folds.query_ids):
        raise RuntimeError("donor and target OOF order differs")
    source = {
        "ad": np.load(paths.base_dir / "official_spectra" / "angle_delay.npy", mmap_mode="r"),
        "pas": np.load(paths.base_dir / "official_spectra" / "pas.npy", mmap_mode="r"),
        "pdp": np.load(paths.base_dir / "official_spectra" / "pdp.npy", mmap_mode="r"),
    }
    axes: dict[str, int | tuple[int, ...]] = {"ad": (0, 1, 2), "pas": 0, "pdp": 2}
    weight_by_domain = {"ad": pas_weights, "pas": pas_weights, "pdp": pdp_weights}
    for name, output in outputs.items():
        shape = (len(query_ids), *source[name].shape[1:])
        target, partial = _atomic_memmap(output, np.dtype(np.float16), shape)
        for row in range(len(query_ids)):
            target[row] = spectral_target(
                source[name], donor_ids[row], weight_by_domain[name][row], axes[name]
            ).astype(np.float16)
            if (row + 1) % 25 == 0 or row + 1 == len(query_ids):
                target.flush()
                print(f"OOF spectral target {name} {row + 1}/{len(query_ids)}", flush=True)
        del target
        os.replace(partial, output)
    return outputs


def _candidate_names() -> list[str]:
    return [
        f"exact_{mode}_{order}_{action.name}_c{cycle:02d}"
        for mode in TARGET_MODES
        for action in ACTIONS
        for order in ORDERS
        for cycle in CHECKPOINTS
    ]


def _cosine(
    left: torch.Tensor, right: torch.Tensor, dims: int | tuple[int, ...]
) -> torch.Tensor:
    numerator = (left * right).sum(dim=dims, dtype=torch.float64)
    denominator = torch.sqrt(
        left.square().sum(dim=dims, dtype=torch.float64)
        * right.square().sum(dim=dims, dtype=torch.float64)
    )
    return numerator / torch.where(denominator == 0.0, 1e-12, denominator)


def _metric_components(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
) -> dict[str, np.ndarray]:
    shaped = prediction.reshape(len(prediction), 2, 16, 8, 4, 192)
    prediction_pas = torch.fft.fft2(shaped, dim=(2, 3)).abs().square().float()
    prediction_pdp = torch.fft.ifft(prediction, dim=-1).abs().square().float()
    pas = _cosine(prediction_pas, truth_pas, (1, 2, 3)).mean(dim=(1, 2))
    pdp = _cosine(prediction_pdp, truth_pdp, 3).mean(dim=(1, 2))
    error = (prediction - truth).abs().square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    truth_energy = truth.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    predicted_energy = prediction.abs().square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    cross = (prediction * truth.conj()).sum(
        dim=(1, 2, 3), dtype=torch.complex128
    ).real
    return {
        "pas": pas.cpu().numpy(),
        "pdp": pdp.cpu().numpy(),
        "error": error.cpu().numpy(),
        "truth_energy": truth_energy.cpu().numpy(),
        "predicted_energy": predicted_energy.cpu().numpy(),
        "cross": cross.cpu().numpy(),
    }


def generate_projection_metrics(
    paths: RuntimePaths,
    name: str,
    base_path: Path,
    targets: dict[str, Path],
    folds: FoldMetadata,
    device: torch.device,
    config: SpectralAssetConfig,
    force: bool,
) -> dict[str, np.ndarray]:
    output = projection_metrics_path(paths, name)
    required = {
        "pas",
        "pdp",
        "error",
        "truth_energy",
        "predicted_energy",
        "cross",
        "candidate_names",
        "query_ids",
        "fold_id",
        "site",
    }
    if output.is_file() and not force:
        with np.load(output, allow_pickle=False) as loaded:
            if required.issubset(loaded.files) and np.array_equal(
                loaded["query_ids"], folds.query_ids
            ):
                return {key: np.array(loaded[key], copy=True) for key in loaded.files}
    names = _candidate_names()
    name_to_index = {candidate: index for index, candidate in enumerate(names)}
    result = {
        key: np.empty((len(names), len(folds.query_ids)), dtype=np.float64)
        for key in ("pas", "pdp", "error", "truth_energy", "predicted_energy", "cross")
    }
    base = np.load(base_path, mmap_mode="r")
    ad = np.load(targets["ad"], mmap_mode="r")
    raw_pas_source = np.load(targets["pas"], mmap_mode="r")
    raw_pdp_source = np.load(targets["pdp"], mmap_mode="r")
    grouped_pas = np.load(oof_grouped_pas_path(paths), mmap_mode="r")
    grouped_pdp = np.load(oof_grouped_pdp_path(paths), mmap_mode="r")
    truth_all = load_official_data(paths.data_dir).train_channel(mmap_mode="r")
    for start in range(0, len(folds.query_ids), config.metric_batch_size):
        end = min(start + config.metric_batch_size, len(folds.query_ids))
        raw_pas, raw_pdp = globally_shrunk_targets(
            np.asarray(raw_pas_source[start:end], dtype=np.float32),
            np.asarray(raw_pdp_source[start:end], dtype=np.float32),
        )
        calibrated_pas = blend_group_pas(
            raw_pas,
            np.asarray(grouped_pas[start:end], dtype=np.float32),
            config.pas_group_blend,
        )
        calibrated_pdp = blend_group_pdp(
            raw_pdp,
            np.asarray(grouped_pdp[start:end], dtype=np.float32),
            config.pdp_group_blend,
        )
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(
                truth_all[folds.query_ids[start:end]], dtype=np.complex64, copy=True
            ),
            device=device,
        )
        truth_shaped = truth.reshape(end - start, 2, 16, 8, 4, 192)
        truth_pas = torch.fft.fft2(truth_shaped, dim=(2, 3)).abs().square().float()
        truth_pdp = torch.fft.ifft(truth, dim=-1).abs().square().float()
        ad_tensor = torch.as_tensor(
            np.asarray(ad[start:end], dtype=np.float32), device=device
        )
        raw_pas_tensor = torch.as_tensor(raw_pas, device=device)
        raw_pdp_tensor = torch.as_tensor(raw_pdp, device=device)
        calibrated_pas_tensor = torch.as_tensor(calibrated_pas, device=device)
        calibrated_pdp_tensor = torch.as_tensor(calibrated_pdp, device=device)
        with torch.inference_mode():
            for mode in TARGET_MODES:
                local_pas, local_pdp = mode_targets(
                    mode,
                    raw_pas_tensor,
                    raw_pdp_tensor,
                    calibrated_pas_tensor,
                    calibrated_pdp_tensor,
                )
                for action_index, action in enumerate(ACTIONS):
                    for order in ORDERS:
                        candidate, desired_pas, desired_pdp, reference_power = (
                            initialize_trajectory(
                                base_tensor,
                                ad_tensor,
                                local_pas,
                                local_pdp,
                                action_index,
                                order,
                            )
                        )
                        for cycle in range(1, max(CHECKPOINTS) + 1):
                            if cycle > 1:
                                candidate = continue_trajectory(
                                    candidate,
                                    desired_pas,
                                    desired_pdp,
                                    reference_power,
                                    order,
                                )
                            if cycle not in CHECKPOINTS:
                                continue
                            candidate_name = (
                                f"exact_{mode}_{order}_{action.name}_c{cycle:02d}"
                            )
                            index = name_to_index[candidate_name]
                            values = _metric_components(
                                candidate, truth, truth_pas, truth_pdp
                            )
                            for key, value in values.items():
                                result[key][index, start:end] = value
        print(f"projection metrics {name} {end}/{len(folds.query_ids)}", flush=True)
    data = load_official_data(paths.data_dir)
    result["candidate_names"] = np.asarray(names)
    result["query_ids"] = folds.query_ids.astype(np.int32)
    result["fold_id"] = folds.fold_id
    result["site"] = data.train_sites[folds.query_ids].astype(np.int8)
    atomic_npz(output, **result)
    return result


def _optimal_scale(predicted_energy: np.ndarray, cross: np.ndarray, mask: np.ndarray) -> float:
    denominator = float(np.sum(predicted_energy[mask], dtype=np.float64))
    numerator = float(np.sum(cross[mask], dtype=np.float64))
    if denominator <= TINY:
        return 1.0
    return float(np.clip(numerator / denominator, 0.05, 1.50))


def _scaled_error(
    predicted_energy: np.ndarray,
    truth_energy: np.ndarray,
    cross: np.ndarray,
    scale: float | np.ndarray,
) -> np.ndarray:
    value = np.asarray(scale) ** 2 * predicted_energy + truth_energy - 2.0 * np.asarray(scale) * cross
    return np.maximum(value, 0.0)


def _statistics(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    selected = np.ones(len(pas), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    count = int(np.sum(selected))
    pas_mean = float(np.mean(pas[selected], dtype=np.float64))
    pdp_mean = float(np.mean(pdp[selected], dtype=np.float64))
    nmse = float(np.sum(error[selected], dtype=np.float64) / max(np.sum(energy[selected], dtype=np.float64), TINY))
    nmse_score = 1.0 / (1.0 + nmse)
    return {
        "count": count,
        "pas": pas_mean,
        "pdp": pdp_mean,
        "nmse": nmse,
        "nmse_score": nmse_score,
        "score": 0.4 * pas_mean + 0.4 * pdp_mean + 0.2 * nmse_score,
    }


def select_projection_policy(metrics: dict[str, np.ndarray]) -> dict[str, Any]:
    names = metrics["candidate_names"].astype(str)
    fold_id = metrics["fold_id"].astype(np.int8)
    site = metrics["site"].astype(np.int8)
    count = len(fold_id)
    selected_candidate = np.empty(count, dtype=np.int16)
    selected_scale = np.empty(count, dtype=np.float32)
    decisions: list[dict[str, Any]] = []
    for held_fold in range(5):
        for site_value in (0, 1):
            train = (fold_id != held_fold) & (site == site_value)
            held = (fold_id == held_fold) & (site == site_value)
            if not np.any(held):
                continue
            candidates = []
            for index in range(len(names)):
                scale = _optimal_scale(
                    metrics["predicted_energy"][index], metrics["cross"][index], train
                )
                error = _scaled_error(
                    metrics["predicted_energy"][index],
                    metrics["truth_energy"][index],
                    metrics["cross"][index],
                    scale,
                )
                score = float(
                    _statistics(
                        metrics["pas"][index],
                        metrics["pdp"][index],
                        error,
                        metrics["truth_energy"][index],
                        train,
                    )["score"]
                )
                candidates.append((score, index, scale))
            training_score, index, scale = max(candidates)
            selected_candidate[held] = index
            selected_scale[held] = scale
            decisions.append(
                {
                    "held_fold": held_fold,
                    "site": site_value,
                    "candidate": str(names[index]),
                    "candidate_index": int(index),
                    "scale": float(scale),
                    "training_score": training_score,
                    "held_count": int(np.sum(held)),
                }
            )
    point = np.arange(count)
    selected_error = _scaled_error(
        metrics["predicted_energy"][selected_candidate, point],
        metrics["truth_energy"][selected_candidate, point],
        metrics["cross"][selected_candidate, point],
        selected_scale,
    )
    overall = _statistics(
        metrics["pas"][selected_candidate, point],
        metrics["pdp"][selected_candidate, point],
        selected_error,
        metrics["truth_energy"][selected_candidate, point],
    )
    fold_scores = [
        {
            "fold": fold,
            **_statistics(
                metrics["pas"][selected_candidate, point],
                metrics["pdp"][selected_candidate, point],
                selected_error,
                metrics["truth_energy"][selected_candidate, point],
                fold_id == fold,
            ),
        }
        for fold in range(5)
    ]
    final_policy = []
    for site_value in (0, 1):
        mask = site == site_value
        candidates = []
        for index in range(len(names)):
            scale = _optimal_scale(
                metrics["predicted_energy"][index], metrics["cross"][index], mask
            )
            error = _scaled_error(
                metrics["predicted_energy"][index],
                metrics["truth_energy"][index],
                metrics["cross"][index],
                scale,
            )
            score = float(
                _statistics(
                    metrics["pas"][index],
                    metrics["pdp"][index],
                    error,
                    metrics["truth_energy"][index],
                    mask,
                )["score"]
            )
            candidates.append((score, index, scale))
        score, index, scale = max(candidates)
        final_policy.append(
            {
                "site": site_value,
                "candidate_index": int(index),
                "candidate": str(names[index]),
                "scale": float(scale),
                "oof_site_score": score,
            }
        )
    return {
        "nested_selected": overall,
        "nested_folds": fold_scores,
        "nested_decisions": decisions,
        "final_policy": final_policy,
        "selected_candidate": selected_candidate,
        "selected_scale": selected_scale,
    }


def _target_batch(
    paths: RuntimePaths,
    targets: dict[str, Path],
    rows: np.ndarray,
    config: SpectralAssetConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ad = np.asarray(np.load(targets["ad"], mmap_mode="r")[rows], dtype=np.float32)
    raw_pas, raw_pdp = globally_shrunk_targets(
        np.asarray(np.load(targets["pas"], mmap_mode="r")[rows], dtype=np.float32),
        np.asarray(np.load(targets["pdp"], mmap_mode="r")[rows], dtype=np.float32),
    )
    grouped_pas = np.asarray(
        np.load(oof_grouped_pas_path(paths), mmap_mode="r")[rows], dtype=np.float32
    )
    grouped_pdp = np.asarray(
        np.load(oof_grouped_pdp_path(paths), mmap_mode="r")[rows], dtype=np.float32
    )
    calibrated_pas = blend_group_pas(raw_pas, grouped_pas, config.pas_group_blend)
    calibrated_pdp = blend_group_pdp(raw_pdp, grouped_pdp, config.pdp_group_blend)
    return ad, raw_pas, raw_pdp, calibrated_pas, calibrated_pdp


def _apply_named_projection(
    base: torch.Tensor,
    ad: torch.Tensor,
    raw_pas: torch.Tensor,
    raw_pdp: torch.Tensor,
    calibrated_pas: torch.Tensor,
    calibrated_pdp: torch.Tensor,
    candidate_name: str,
) -> torch.Tensor:
    match = EXACT_PATTERN.fullmatch(candidate_name)
    if match is None:
        raise ValueError(f"unsupported projection candidate: {candidate_name}")
    mode, order, action_name, cycles_text = match.groups()
    action_index = next(
        index for index, action in enumerate(ACTIONS) if action.name == action_name
    )
    pas, pdp = mode_targets(
        mode, raw_pas, raw_pdp, calibrated_pas, calibrated_pdp
    )
    candidate, desired_pas, desired_pdp, reference_power = initialize_trajectory(
        base, ad, pas, pdp, action_index, order
    )
    for _ in range(2, int(cycles_text) + 1):
        candidate = continue_trajectory(
            candidate, desired_pas, desired_pdp, reference_power, order
        )
    return candidate


def write_nested_projection(
    paths: RuntimePaths,
    base_path: Path,
    targets: dict[str, Path],
    selection: dict[str, Any],
    output: Path,
    device: torch.device,
    config: SpectralAssetConfig,
    force: bool,
) -> Path:
    if output.is_file() and not force:
        value = np.load(output, mmap_mode="r")
        if value.shape == (len(selection["selected_candidate"]), 256, 4, 192) and value.dtype == np.complex64:
            return output
    base = np.load(base_path, mmap_mode="r")
    names = _candidate_names()
    selected = np.asarray(selection["selected_candidate"], dtype=np.int64)
    scales = np.asarray(selection["selected_scale"], dtype=np.float32)
    target, partial = _atomic_memmap(
        output, np.dtype(np.complex64), (len(selected), 256, 4, 192)
    )
    for candidate_index in np.unique(selected):
        rows = np.flatnonzero(selected == candidate_index)
        for start in range(0, len(rows), config.metric_batch_size):
            local = rows[start : start + config.metric_batch_size]
            ad, raw_pas, raw_pdp, calibrated_pas, calibrated_pdp = _target_batch(
                paths, targets, local, config
            )
            with torch.inference_mode():
                prediction = _apply_named_projection(
                    torch.as_tensor(
                        np.array(base[local], dtype=np.complex64, copy=True), device=device
                    ),
                    torch.as_tensor(ad, device=device),
                    torch.as_tensor(raw_pas, device=device),
                    torch.as_tensor(raw_pdp, device=device),
                    torch.as_tensor(calibrated_pas, device=device),
                    torch.as_tensor(calibrated_pdp, device=device),
                    names[int(candidate_index)],
                )
            target[local] = (
                prediction.cpu().numpy() * scales[local, None, None, None]
            ).astype(np.complex64)
        target.flush()
    del target
    os.replace(partial, output)
    return output


def train_spectral_assets(
    paths: RuntimePaths,
    pipeline_config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool = False,
    training_config: SpectralAssetConfig | None = None,
) -> dict[str, Any]:
    """Fit group calibration plus reference/final exact-projection policies."""

    started = time.time()
    config = training_config or SpectralAssetConfig(
        metric_batch_size=max(1, min(batch_size, 4))
    )
    _, layout, preparation = prepare_training_inputs(
        paths, pipeline_config, device, batch_size, force=False
    )
    folds = load_fold_metadata(layout.fold_cache, layout.fold_metadata)
    group_report = build_group_calibration(paths, folds, device, config, force)
    targets = build_oof_spectral_targets(paths, folds, force)
    reference_metrics = generate_projection_metrics(
        paths,
        "reference",
        oof_foundation_path(paths),
        targets,
        folds,
        device,
        config,
        force,
    )
    final_metrics = generate_projection_metrics(
        paths,
        "final",
        oof_mixture_path(paths),
        targets,
        folds,
        device,
        config,
        force,
    )
    reference_selection = select_projection_policy(reference_metrics)
    final_selection = select_projection_policy(final_metrics)
    reference_output = write_nested_projection(
        paths,
        oof_foundation_path(paths),
        targets,
        reference_selection,
        oof_reference_projection_path(paths),
        device,
        config,
        force,
    )
    final_output = write_nested_projection(
        paths,
        oof_mixture_path(paths),
        targets,
        final_selection,
        oof_final_projection_path(paths),
        device,
        config,
        force,
    )
    policy = {
        "experiment": "Round2-V17-official-retraining",
        "reference_policy": reference_selection["final_policy"],
        "final_policy": final_selection["final_policy"],
        "reference_nested_selected": reference_selection["nested_selected"],
        "reference_nested_folds": reference_selection["nested_folds"],
        "reference_nested_decisions": reference_selection["nested_decisions"],
        "final_nested_selected": final_selection["nested_selected"],
        "final_nested_folds": final_selection["nested_folds"],
        "final_nested_decisions": final_selection["nested_decisions"],
    }
    atomic_json(projection_policy_path(paths), policy)
    report = {
        "stage": "train_spectral_assets",
        "status": "complete",
        "config": asdict(config),
        "preparation": preparation,
        "group_calibration": group_report,
        "targets": {name: portable_path(path) for name, path in targets.items()},
        "reference": {
            "metrics": portable_path(projection_metrics_path(paths, "reference")),
            "selection": {
                key: value
                for key, value in reference_selection.items()
                if key not in ("selected_candidate", "selected_scale")
            },
            "output": portable_path(reference_output),
        },
        "final": {
            "metrics": portable_path(projection_metrics_path(paths, "final")),
            "selection": {
                key: value
                for key, value in final_selection.items()
                if key not in ("selected_candidate", "selected_scale")
            },
            "output": portable_path(final_output),
        },
        "policy": portable_path(projection_policy_path(paths)),
        "wall_seconds": float(time.time() - started),
        "device": str(device),
        "peak_torch_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
        "measured_bottleneck": "280-candidate exact-projection fixed500 sweep",
    }
    atomic_json(paths.report_dir / "train_spectral_assets.json", report)
    return report

