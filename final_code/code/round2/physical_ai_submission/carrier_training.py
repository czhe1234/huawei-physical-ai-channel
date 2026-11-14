"""Build strict OOF complex carriers and fit the production carrier ridge stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from .base_runtime import carrier_bank
from .base_runtime.carrier_phase import estimate_carrier
from .base_runtime.spatial_baselines import batched_predict_many_torch
from .config import PipelineConfig, RuntimePaths
from .data import load_official_data
from .io_utils import atomic_json, atomic_npy, atomic_npz, portable_path
from .training import prepare_training_inputs
from .training_common import FoldMetadata, load_fold_metadata


TINY = 1e-30


@dataclass(frozen=True)
class CarrierTrainingConfig:
    carrier_pairs: int = 500
    gpu_batch_size: int = 8
    ridges: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    sum_penalty: float = 0.0
    seed: int = 20_260_805


def _root(paths: RuntimePaths) -> Path:
    return paths.work_dir / "training" / "carrier"


def oof_bank_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "oof_carrier_bank.npy"


def _fold_bank_path(paths: RuntimePaths, fold: int) -> Path:
    return _root(paths) / "fold_banks" / f"fold_{fold}.npy"


def _power_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "train_point_power.npy"


def gram_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "oof_port_gram.npz"


def oof_foundation_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "oof_foundation.npy"


def oof_mixture_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "oof_carrier_mixture.npy"


def carrier_weights_path(paths: RuntimePaths) -> Path:
    return paths.model_dir / "base_field" / "carrier_mix_weights.npz"


def _valid_bank(path: Path, count: int, carriers: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, mmap_mode="r")
        return value.shape == (carriers, count, 256, 4, 192) and value.dtype == np.complex64
    except (OSError, ValueError):
        return False


def _point_power(paths: RuntimePaths, channel: np.ndarray) -> np.ndarray:
    path = _power_path(paths)
    if path.is_file():
        value = np.load(path, allow_pickle=False)
        if value.shape == (len(channel),):
            return value.astype(np.float64, copy=False)
    value = np.empty(len(channel), dtype=np.float64)
    for start in range(0, len(channel), 16):
        end = min(start + 16, len(channel))
        block = np.asarray(channel[start:end], dtype=np.complex64)
        value[start:end] = np.sum(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64)
    atomic_npy(path, value)
    return value


def _site_neighbors(
    train_positions: np.ndarray,
    train_sites: np.ndarray,
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.empty((len(query_ids), max_k), dtype=np.float64)
    neighbors = np.empty((len(query_ids), max_k), dtype=np.int64)
    for site in np.unique(train_sites[query_ids]):
        rows = np.flatnonzero(train_sites[query_ids] == site)
        bank = fit_ids[train_sites[fit_ids] == site]
        local_distance, local = cKDTree(train_positions[bank, :2]).query(
            train_positions[query_ids[rows], :2], k=max_k, workers=-1
        )
        distances[rows] = local_distance
        neighbors[rows] = bank[local]
    return distances, neighbors


def _write_fold_bank(
    paths: RuntimePaths,
    fold: int,
    folds: FoldMetadata,
    channel: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
    point_power: np.ndarray,
    device_channel: torch.Tensor,
    config: CarrierTrainingConfig,
    force: bool,
) -> dict[str, Any]:
    templates = carrier_bank.carrier_configs()
    query_ids = folds.query_indices[fold]
    output = _fold_bank_path(paths, fold)
    if _valid_bank(output, len(query_ids), len(templates)) and not force:
        return {
            "fold": fold,
            "status": "reused",
            "query_count": int(len(query_ids)),
            "output": portable_path(output),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    target = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.complex64,
        shape=(len(templates), len(query_ids), 256, 4, 192),
    )
    fit_ids = folds.fit_indices[fold]
    carrier = estimate_carrier(
        channel,
        positions,
        sites,
        bs_positions,
        fit_ids,
        point_power,
        max_pairs_per_site=config.carrier_pairs,
        seed=config.seed + fold * 101,
    )
    effective_by_site = {
        int(site): [
            replace(template, beta=value.beta, gamma=value.gamma)
            for template in templates
        ]
        for site, value in carrier.items()
    }
    max_k = max(template.k for template in templates)
    distances, neighbors = _site_neighbors(
        positions, sites, fit_ids, query_ids, max_k
    )
    for start in range(0, len(query_ids), config.gpu_batch_size):
        end = min(start + config.gpu_batch_size, len(query_ids))
        local_query = query_ids[start:end]
        prediction, _, _ = batched_predict_many_torch(
            device_channel,
            neighbors[start:end],
            positions[neighbors[start:end]],
            positions[local_query],
            distances[start:end],
            bs_positions,
            sites[local_query],
            effective_by_site,
            point_power,
        )
        target[:, start:end] = prediction.permute(1, 0, 2, 3, 4).cpu().numpy()
        target.flush()
        print(f"carrier OOF fold={fold} {end}/{len(query_ids)}", flush=True)
        del prediction
    target.flush()
    del target
    os.replace(partial, output)
    return {
        "fold": fold,
        "status": "complete",
        "fit_count": int(len(fit_ids)),
        "query_count": int(len(query_ids)),
        "output": portable_path(output),
        "carrier": {str(site): value.to_dict() for site, value in carrier.items()},
    }


def build_oof_carrier_bank(
    paths: RuntimePaths,
    folds: FoldMetadata,
    device: torch.device,
    config: CarrierTrainingConfig,
    force: bool,
) -> tuple[Path, list[dict[str, Any]]]:
    data = load_official_data(paths.data_dir)
    templates = carrier_bank.carrier_configs()
    output = oof_bank_path(paths)
    if _valid_bank(output, len(folds.query_ids), len(templates)) and not force:
        return output, [{"status": "reused", "output": portable_path(output)}]
    channel = data.train_channel(mmap_mode="r")
    point_power = _point_power(paths, channel)
    device_channel = torch.as_tensor(
        np.asarray(channel), dtype=torch.complex64, device=device
    )
    fold_reports = []
    for fold in range(5):
        fold_reports.append(
            _write_fold_bank(
                paths,
                fold,
                folds,
                channel,
                data.train_positions,
                data.train_sites,
                data.setup.bs_positions,
                point_power,
                device_channel,
                config,
                force,
            )
        )
    del device_channel
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    combined = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.complex64,
        shape=(len(templates), len(folds.query_ids), 256, 4, 192),
    )
    lookup = folds.query_lookup
    for fold in range(5):
        rows = lookup[folds.query_indices[fold]]
        source = np.load(_fold_bank_path(paths, fold), mmap_mode="r")
        for start in range(0, len(rows), 8):
            end = min(start + 8, len(rows))
            combined[:, rows[start:end]] = source[:, start:end]
        combined.flush()
    del combined
    os.replace(partial, output)
    return output, fold_reports


def _build_foundation(
    paths: RuntimePaths,
    data_sites: np.ndarray,
    query_ids: np.ndarray,
    bank_path: Path,
    force: bool,
) -> Path:
    output = oof_foundation_path(paths)
    if output.is_file() and not force:
        value = np.load(output, mmap_mode="r")
        if value.shape == (len(query_ids), 256, 4, 192) and value.dtype == np.complex64:
            return output
    names = [config.name for config in carrier_bank.carrier_configs()]
    selected = {
        0: names.index("affine_k24_steer_axis90_v8pi"),
        1: names.index("radial_idw_k8"),
    }
    bank = np.load(bank_path, mmap_mode="r")
    partial = output.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    target = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.complex64,
        shape=(len(query_ids), 256, 4, 192),
    )
    query_sites = data_sites[query_ids]
    for site, carrier_index in selected.items():
        rows = np.flatnonzero(query_sites == site)
        for start in range(0, len(rows), 8):
            local = rows[start : start + 8]
            target[local] = bank[carrier_index, local]
        target.flush()
    del target
    os.replace(partial, output)
    return output


def _compute_gram(
    paths: RuntimePaths,
    bank_path: Path,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    device: torch.device,
    force: bool,
) -> dict[str, np.ndarray]:
    output = gram_path(paths)
    required = {"gram", "cross", "energy", "query_ids", "fold_id"}
    if output.is_file() and not force:
        with np.load(output, allow_pickle=False) as loaded:
            if required.issubset(loaded.files) and np.array_equal(
                loaded["query_ids"], query_ids
            ):
                return {name: np.array(loaded[name], copy=True) for name in loaded.files}
    bank = np.load(bank_path, mmap_mode="r")
    truth = load_official_data(paths.data_dir).train_channel(mmap_mode="r")
    carriers = len(bank)
    gram = np.empty((len(query_ids), 8, carriers, carriers), dtype=np.complex64)
    cross = np.empty((len(query_ids), 8, carriers), dtype=np.complex64)
    energy = np.empty((len(query_ids), 8), dtype=np.float64)
    for start in range(0, len(query_ids), 2):
        end = min(start + 2, len(query_ids))
        count = end - start
        candidate = torch.as_tensor(
            np.array(bank[:, start:end], dtype=np.complex64, copy=True), device=device
        ).permute(1, 0, 2, 3, 4)
        candidate = candidate.reshape(count, carriers, 2, 16, 8, 4, 192)
        candidate = candidate.permute(0, 2, 5, 1, 3, 4, 6).reshape(
            count, 8, carriers, -1
        )
        target = torch.as_tensor(
            np.array(truth[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        ).reshape(count, 2, 16, 8, 4, 192)
        target = target.permute(0, 1, 4, 2, 3, 5).reshape(count, 8, -1)
        with torch.inference_mode():
            gram[start:end] = torch.einsum(
                "bgcd,bged->bgce", candidate.conj(), candidate
            ).cpu().numpy()
            cross[start:end] = torch.einsum(
                "bgcd,bgd->bgc", candidate.conj(), target
            ).cpu().numpy()
            energy[start:end] = target.abs().square().sum(
                dim=2, dtype=torch.float64
            ).cpu().numpy()
        print(f"carrier Gram {end}/{len(query_ids)}", flush=True)
    atomic_npz(
        output,
        gram=gram,
        cross=cross,
        energy=energy,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
    )
    return {
        "gram": gram,
        "cross": cross,
        "energy": energy,
        "query_ids": query_ids,
        "fold_id": fold_id,
    }


def _solve_weights(
    gram: np.ndarray, cross: np.ndarray, ridge: float, sum_penalty: float
) -> np.ndarray:
    carriers = gram.shape[-1]
    scale = max(float(np.trace(gram).real / carriers), TINY)
    system = np.asarray(gram, dtype=np.complex128).copy()
    right = np.asarray(cross, dtype=np.complex128).copy()
    system += np.eye(carriers, dtype=np.complex128) * ridge * scale
    if sum_penalty > 0.0:
        penalty = sum_penalty * scale
        system += penalty * np.ones((carriers, carriers), dtype=np.complex128)
        right += penalty * np.ones(carriers, dtype=np.complex128)
    try:
        return np.linalg.solve(system, right).astype(np.complex64)
    except np.linalg.LinAlgError:
        return (np.linalg.pinv(system) @ right).astype(np.complex64)


def _fit_weights(
    cache: dict[str, np.ndarray],
    rows: np.ndarray,
    site: np.ndarray,
    ridge: float,
    sum_penalty: float,
) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for site_value in np.unique(site):
        local = rows[site[rows] == site_value]
        weights = []
        for port in range(8):
            total_gram = cache["gram"][local, port].sum(axis=0, dtype=np.complex128)
            total_cross = cache["cross"][local, port].sum(axis=0, dtype=np.complex128)
            weights.append(
                _solve_weights(total_gram, total_cross, ridge, sum_penalty)
            )
        output[int(site_value)] = np.stack(weights)
    return output


def _point_error(
    cache: dict[str, np.ndarray],
    rows: np.ndarray,
    site: np.ndarray,
    weights: dict[int, np.ndarray],
) -> tuple[float, float]:
    total_error = 0.0
    total_energy = float(cache["energy"][rows].sum(dtype=np.float64))
    for row in rows:
        local = weights[int(site[row])]
        for port in range(8):
            weight = local[port].astype(np.complex128)
            value = (
                np.vdot(
                    weight,
                    cache["gram"][row, port].astype(np.complex128) @ weight,
                ).real
                - 2.0
                * np.vdot(
                    weight, cache["cross"][row, port].astype(np.complex128)
                ).real
                + cache["energy"][row, port]
            )
            total_error += max(float(value), 0.0)
    return total_error, total_energy


def _select_ridge(
    cache: dict[str, np.ndarray],
    site: np.ndarray,
    config: CarrierTrainingConfig,
) -> tuple[list[dict[str, Any]], float, list[dict[str, Any]]]:
    fold_id = cache["fold_id"].astype(np.int8)
    outer_records: list[dict[str, Any]] = []
    nested_choices: list[dict[str, Any]] = []
    rows = np.arange(len(fold_id), dtype=np.int64)
    for outer in range(5):
        outer_train = rows[fold_id != outer]
        held = rows[fold_id == outer]
        candidate_errors = np.zeros(len(config.ridges), dtype=np.float64)
        candidate_energy = np.zeros(len(config.ridges), dtype=np.float64)
        for inner in range(5):
            if inner == outer:
                continue
            fit = outer_train[fold_id[outer_train] != inner]
            valid = outer_train[fold_id[outer_train] == inner]
            for index, ridge in enumerate(config.ridges):
                weights = _fit_weights(cache, fit, site, ridge, config.sum_penalty)
                error, energy = _point_error(cache, valid, site, weights)
                candidate_errors[index] += error
                candidate_energy[index] += energy
        nmse = candidate_errors / np.maximum(candidate_energy, TINY)
        selected_index = int(np.argmin(nmse))
        ridge = config.ridges[selected_index]
        weights = _fit_weights(cache, outer_train, site, ridge, config.sum_penalty)
        error, energy = _point_error(cache, held, site, weights)
        outer_records.append(
            {
                "fold": outer,
                "ridge": ridge,
                "inner_nmse": float(nmse[selected_index]),
                "held_nmse": float(error / max(energy, TINY)),
            }
        )
        nested_choices.append(
            {"fold": outer, "ridge": ridge, "weights": weights}
        )

    posterior = []
    for ridge in config.ridges:
        error = 0.0
        energy = 0.0
        fold_nmse = []
        for held_fold in range(5):
            fit = rows[fold_id != held_fold]
            held = rows[fold_id == held_fold]
            weights = _fit_weights(cache, fit, site, ridge, config.sum_penalty)
            local_error, local_energy = _point_error(cache, held, site, weights)
            error += local_error
            energy += local_energy
            fold_nmse.append(float(local_error / max(local_energy, TINY)))
        posterior.append(
            {
                "ridge": ridge,
                "nmse": float(error / max(energy, TINY)),
                "fold_nmse": fold_nmse,
            }
        )
    selected_ridge = float(min(posterior, key=lambda row: row["nmse"])["ridge"])
    return outer_records, selected_ridge, nested_choices


def _mix_rows(
    bank: np.ndarray,
    rows: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    block = np.asarray(bank[:, rows], dtype=np.complex64)
    candidates = (
        block.reshape(17, len(rows), 2, 16, 8, 4, 192)
        .transpose(1, 2, 5, 0, 3, 4, 6)
        .reshape(len(rows), 8, 17, 16, 8, 192)
    )
    mixed = np.einsum("bgcvhd,gc->bgvhd", candidates, weights, optimize=True)
    return (
        mixed.reshape(len(rows), 2, 4, 16, 8, 192)
        .transpose(0, 1, 3, 4, 2, 5)
        .reshape(len(rows), 256, 4, 192)
        .astype(np.complex64)
    )


def _write_oof_mixture(
    paths: RuntimePaths,
    bank_path: Path,
    fold_id: np.ndarray,
    site: np.ndarray,
    nested_choices: list[dict[str, Any]],
    force: bool,
) -> Path:
    output = oof_mixture_path(paths)
    if output.is_file() and not force:
        value = np.load(output, mmap_mode="r")
        if value.shape == (len(fold_id), 256, 4, 192) and value.dtype == np.complex64:
            return output
    bank = np.load(bank_path, mmap_mode="r")
    partial = output.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    target = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.complex64,
        shape=(len(fold_id), 256, 4, 192),
    )
    by_fold = {int(row["fold"]): row["weights"] for row in nested_choices}
    for fold in range(5):
        for site_value in np.unique(site):
            rows = np.flatnonzero((fold_id == fold) & (site == site_value))
            if not len(rows):
                continue
            weights = by_fold[fold][int(site_value)]
            for start in range(0, len(rows), 4):
                local = rows[start : start + 4]
                target[local] = _mix_rows(bank, local, weights)
            target.flush()
    del target
    os.replace(partial, output)
    return output


def train_carrier_assets(
    paths: RuntimePaths,
    pipeline_config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool = False,
    training_config: CarrierTrainingConfig | None = None,
) -> dict[str, Any]:
    """Generate the OOF carrier bank, strict mixture, and production weights."""

    started = time.time()
    config = training_config or CarrierTrainingConfig(
        gpu_batch_size=max(1, min(batch_size, 16))
    )
    data, layout, preparation = prepare_training_inputs(
        paths, pipeline_config, device, batch_size, force=False
    )
    folds = load_fold_metadata(layout.fold_cache, layout.fold_metadata)
    bank, fold_reports = build_oof_carrier_bank(
        paths, folds, device, config, force
    )
    foundation = _build_foundation(
        paths, data.train_sites, folds.query_ids, bank, force
    )
    cache = _compute_gram(
        paths, bank, folds.query_ids, folds.fold_id, device, force
    )
    query_site = data.train_sites[folds.query_ids].astype(np.int8)
    nested, selected_ridge, nested_choices = _select_ridge(
        cache, query_site, config
    )
    mixture = _write_oof_mixture(
        paths, bank, folds.fold_id, query_site, nested_choices, force
    )
    rows = np.arange(len(folds.query_ids), dtype=np.int64)
    production_weights = _fit_weights(
        cache, rows, query_site, selected_ridge, config.sum_penalty
    )
    asset = carrier_weights_path(paths)
    atomic_npz(
        asset,
        **{
            f"label_{site}": value.astype(np.complex64)
            for site, value in production_weights.items()
        },
        labels=np.asarray(sorted(production_weights), dtype=np.int16),
        scope=np.asarray("site"),
        ports=np.asarray("port8"),
        ridge=np.float64(selected_ridge),
        sum_penalty=np.float64(config.sum_penalty),
    )
    report = {
        "stage": "train_carrier_assets",
        "status": "complete",
        "config": asdict(config),
        "preparation": preparation,
        "fold_banks": fold_reports,
        "bank": portable_path(bank),
        "gram": portable_path(gram_path(paths)),
        "foundation": portable_path(foundation),
        "mixture": portable_path(mixture),
        "nested": nested,
        "selected_ridge": selected_ridge,
        "asset": portable_path(asset),
        "wall_seconds": float(time.time() - started),
        "device": str(device),
        "peak_torch_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
        "measured_bottleneck": "OOF complex carrier synthesis and Gram accumulation",
    }
    atomic_json(paths.report_dir / "train_carrier_assets.json", report)
    return report

