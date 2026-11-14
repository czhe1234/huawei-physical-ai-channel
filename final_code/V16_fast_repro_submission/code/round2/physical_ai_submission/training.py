"""Retrain the PAS/PDP spectral calibrators from official Train inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .base_field import build_official_spectra
from .base_runtime import joint_spectral_model as joint
from .base_runtime import physical_spectral_dataset as spectral_dataset
from .base_runtime import spectral_calibrator as spectral
from .config import PipelineConfig, RuntimePaths
from .data import OfficialData, load_official_data
from .geometry import build_geometry_regime
from .io_utils import atomic_json, atomic_npy, atomic_npz, portable_path
from .training_folds import SpatialHoleFold, build_spatial_hole_folds


DOMAINS = ("pas", "pdp")
COMPONENTS = ("spectral", "joint")
BASE_MODE = "mean"
JOINT_MODE = "joint"

BASE_TRAIN_CONFIG = spectral.TrainConfig(
    epochs=8,
    steps_per_epoch=200,
    pas_batch_size=1024,
    pdp_batch_size=2048,
    validation_batch_size=4096,
    hidden=64,
    context_channels=24,
    blocks=5,
    residual_scale=6.0,
    learning_rate=1.5e-3,
    weight_decay=2e-5,
    hellinger_weight=0.05,
    residual_weight=1e-4,
    seed=20260806,
)

JOINT_TRAIN_CONFIG = spectral.TrainConfig(
    epochs=14,
    steps_per_epoch=240,
    pas_batch_size=512,
    pdp_batch_size=1024,
    validation_batch_size=2048,
    hidden=96,
    context_channels=32,
    blocks=6,
    residual_scale=5.0,
    learning_rate=1e-3,
    weight_decay=5e-5,
    hellinger_weight=0.08,
    residual_weight=2e-4,
    seed=2026080609,
)


@dataclass(frozen=True)
class TrainingLayout:
    visibility: Path
    truth_pas: Path
    truth_pdp: Path
    truth_power: Path
    fold_cache: Path
    fold_metadata: Path
    cache_dir: Path


def parse_selection(
    value: str, allowed: tuple[str, ...], label: str
) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(allowed))
    if not selected or unknown:
        raise ValueError(f"invalid {label}={selected}; allowed={allowed}")
    return selected


def parse_folds(value: str) -> tuple[int, ...]:
    folds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not folds or len(set(folds)) != len(folds):
        raise ValueError("folds must be a non-empty unique subset of 0,1,2,3,4")
    if any(fold < 0 or fold > 4 for fold in folds):
        raise ValueError("folds must be a non-empty unique subset of 0,1,2,3,4")
    return folds


def _physical_stat_path(
    paths: RuntimePaths,
    configuration: str,
    split: str,
    domain: str,
    statistic: str,
) -> Path:
    return (
        paths.work_dir
        / "physical_spectra"
        / split
        / configuration
        / f"{domain}_{statistic}.npy"
    )


def _legacy_positions(data: OfficialData):
    return (
        data.setup,
        data.train_positions,
        data.test_positions,
        data.train_sites,
        data.test_sites,
    )


def _training_layout(paths: RuntimePaths) -> TrainingLayout:
    cache_dir = paths.work_dir / "training" / "spectral_calibrators"
    return TrainingLayout(
        visibility=cache_dir / "visibility_features.npz",
        truth_pas=paths.base_dir / "official_spectra" / "pas.npy",
        truth_pdp=paths.base_dir / "official_spectra" / "pdp.npy",
        truth_power=cache_dir / "official_train_point_power.npy",
        fold_cache=cache_dir / "spatial_hole_folds.npz",
        fold_metadata=cache_dir / "spatial_hole_metadata.npz",
        cache_dir=cache_dir,
    )


def _save_folds(
    layout: TrainingLayout,
    folds: list[SpatialHoleFold],
) -> tuple[np.ndarray, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for fold in folds:
        payload[f"fold{fold.fold}_fit"] = fold.fit_indices.astype(np.int32)
        payload[f"fold{fold.fold}_query"] = fold.query_indices.astype(np.int32)
        payload[f"fold{fold.fold}_ignored"] = fold.ignored_indices.astype(np.int32)
    atomic_npz(layout.fold_cache, **payload)

    query_ids = np.sort(
        np.concatenate([fold.query_indices for fold in folds])
    ).astype(np.int64)
    if len(query_ids) != 500 or len(np.unique(query_ids)) != 500:
        raise RuntimeError(f"training folds must contain 500 unique queries, got {len(query_ids)}")
    row_lookup = np.full(4000, -1, dtype=np.int64)
    row_lookup[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    fold_id = np.full(len(query_ids), -1, dtype=np.int8)
    for fold in folds:
        rows = row_lookup[fold.query_indices]
        if np.any(rows < 0) or np.any(fold_id[rows] >= 0):
            raise RuntimeError(f"invalid or overlapping query IDs in fold {fold.fold}")
        fold_id[rows] = fold.fold
    if np.any(fold_id < 0):
        raise RuntimeError("not every training query was assigned to a fold")
    atomic_npz(
        layout.fold_metadata,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
    )
    return query_ids, fold_id


def _validate_fold_cache(layout: TrainingLayout) -> tuple[np.ndarray, np.ndarray]:
    with np.load(layout.fold_cache, allow_pickle=False) as cached:
        queries = [cached[f"fold{fold}_query"].astype(np.int64) for fold in range(5)]
    query_ids = np.sort(np.concatenate(queries))
    if len(query_ids) != 500 or len(np.unique(query_ids)) != 500:
        raise RuntimeError("cached spatial folds do not contain 500 unique points")
    with np.load(layout.fold_metadata, allow_pickle=False) as metadata:
        stored_ids = metadata["query_ids"].astype(np.int64)
        fold_id = metadata["fold_id"].astype(np.int8)
    if not np.array_equal(query_ids, stored_ids) or fold_id.shape != (500,):
        raise RuntimeError("cached spatial fold metadata is inconsistent")
    return query_ids, fold_id


def _require_train_physical_spectra(paths: RuntimePaths) -> None:
    required = [
        _physical_stat_path(paths, configuration, "train", domain, statistic)
        for configuration in ("base", "scatter")
        for domain in ("pas", "pdp", "ad", "log_power", "path_count")
        for statistic in ("mean", "m2", "count")
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "training physical spectra are incomplete; run prepare-train-prior and "
            f"train-spectra phases first. Missing {len(missing)} files, first={missing[0]}"
        )


def _configure_training_modules(
    paths: RuntimePaths,
    data: OfficialData,
    config: PipelineConfig,
    layout: TrainingLayout,
) -> None:
    def running_path(
        configuration: str, split: str, domain: str, statistic: str
    ) -> Path:
        return _physical_stat_path(paths, configuration, split, domain, statistic)

    load_positions = lambda: _legacy_positions(data)
    for module in (spectral, joint):
        module.GEOMETRY_CACHE = layout.visibility
        module.BASE_CONFIG = "base"
        module.SCATTER_CONFIG = "scatter"
        module.SEEDS = len(config.seeds)
        module.TRUTH_PAS = layout.truth_pas
        module.TRUTH_PDP = layout.truth_pdp
        module.TRUTH_POWER = layout.truth_power
        module.running_path = running_path
        module.load_positions = load_positions

    spectral.ARTIFACT_DIR = paths.model_dir / "base_field" / "spectral"
    spectral.FOLD_CACHE = layout.fold_cache
    spectral.model_path = lambda domain, mode, fold: (
        paths.model_dir / "base_field" / "spectral" / f"{domain}_fold_{fold}.pt"
    )

    joint.ARTIFACT_DIR = layout.cache_dir / "joint"
    joint.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joint.reliability_path = lambda domain, split: (
        joint.ARTIFACT_DIR / f"{split}_{domain}_reliability.npy"
    )
    joint.checkpoint_path = lambda domain, fold: (
        paths.model_dir / "base_field" / "joint" / f"{domain}_fold_{fold}.pt"
    )
    joint.output_path = lambda domain, split: (
        joint.ARTIFACT_DIR / f"{split}_{domain}.npy"
    )

    spectral_dataset.ARTIFACT_DIR = layout.cache_dir
    spectral_dataset.TRUTH_PAS = layout.truth_pas
    spectral_dataset.TRUTH_PDP = layout.truth_pdp
    spectral_dataset.TRUTH_POWER = layout.truth_power
    spectral_dataset.PHYSICAL_ROOT = paths.work_dir / "physical_spectra"
    spectral_dataset.BASE_CONFIG = "base"
    spectral_dataset.SCATTER_CONFIG = "scatter"
    spectral_dataset.SEEDS = len(config.seeds)
    spectral_dataset.running_path = running_path


def prepare_training_inputs(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool = False,
) -> tuple[OfficialData, TrainingLayout, dict[str, Any]]:
    started = time.time()
    _require_train_physical_spectra(paths)
    data = load_official_data(paths.data_dir)
    layout = _training_layout(paths)
    layout.cache_dir.mkdir(parents=True, exist_ok=True)

    official = build_official_spectra(paths, data, device, batch_size, force=force)
    geometry = build_geometry_regime(paths, force=force)
    if force or not layout.visibility.is_file():
        train_visibility = np.asarray(
            geometry["train_visibility"], dtype=np.float32
        )
        atomic_npz(
            layout.visibility,
            names=np.asarray(
                [f"visibility_{index:02d}" for index in range(train_visibility.shape[1])]
            ),
            train=train_visibility,
            test=np.asarray(geometry["test_visibility"], dtype=np.float32),
        )

    with np.load(official["descriptors"], allow_pickle=False) as descriptors:
        total_power = descriptors["total_power"].astype(np.float64)
    if total_power.shape != (data.setup.train_count,):
        raise ValueError(f"invalid official Train power shape: {total_power.shape}")
    if force or not layout.truth_power.is_file():
        atomic_npy(layout.truth_power, total_power)

    if force or not layout.fold_cache.is_file() or not layout.fold_metadata.is_file():
        folds = build_spatial_hole_folds(
            data.train_positions,
            data.test_positions,
            data.train_sites,
            data.test_sites,
        )
        query_ids, fold_id = _save_folds(layout, folds)
    else:
        query_ids, fold_id = _validate_fold_cache(layout)

    _configure_training_modules(paths, data, config, layout)
    report = {
        "stage": "prepare_spectral_training",
        "status": "complete",
        "visibility": portable_path(layout.visibility),
        "truth_pas": portable_path(layout.truth_pas),
        "truth_pdp": portable_path(layout.truth_pdp),
        "truth_power": portable_path(layout.truth_power),
        "fold_cache": portable_path(layout.fold_cache),
        "query_count": int(len(query_ids)),
        "fold_counts": [int(np.sum(fold_id == fold)) for fold in range(5)],
        "wall_seconds": float(time.time() - started),
    }
    atomic_json(paths.report_dir / "prepare_spectral_training.json", report)
    return data, layout, report


def _save_checkpoint(
    path: Path,
    model: spectral.SpectralCalibrator,
    component: str,
    domain: str,
    mode: str,
    fold: int,
    epoch: int,
    config: spectral.TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": f"round2-{component}-spectral-calibrator",
        "domain": domain,
        "mode": mode,
        "fold": int(fold),
        "epoch": int(epoch),
        "config": asdict(config),
        "state_dict": model.state_dict(),
        "context_mean": model.context_mean.detach().cpu(),
        "context_std": model.context_std.detach().cpu(),
    }
    if component == "joint":
        payload["input_channels"] = int(model.stem.in_channels)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _checkpoint_path(paths: RuntimePaths, component: str, domain: str, fold: int) -> Path:
    return paths.model_dir / "base_field" / component / f"{domain}_fold_{fold}.pt"


def _reused_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "status": "reused",
        "checkpoint": portable_path(path),
        "epoch": int(checkpoint["epoch"]),
        "fold": int(checkpoint["fold"]),
        "domain": str(checkpoint["domain"]),
    }


def _train_component(
    paths: RuntimePaths,
    component: str,
    domains: tuple[str, ...],
    folds: tuple[int, ...],
    device: torch.device,
    config: spectral.TrainConfig,
    force: bool,
) -> dict[str, Any]:
    mode = BASE_MODE if component == "spectral" else JOINT_MODE
    result: dict[str, Any] = {}
    for domain_index, domain in enumerate(domains):
        bank = (
            spectral.DomainBank(domain, device)
            if component == "spectral"
            else joint.JointDomainBank(domain, "train", device, force)
        )
        nonzero = np.asarray(bank.nonzero.cpu())
        fold_reports = []
        for fold in folds:
            checkpoint_path = _checkpoint_path(paths, component, domain, fold)
            if checkpoint_path.is_file() and not force:
                fold_reports.append(_reused_checkpoint(checkpoint_path))
                continue

            outer_fit, outer_query, nested_fit, validation = spectral.fold_ids(fold)
            screening_model, best_epoch, history = spectral.train_epochs(
                bank,
                mode,
                nested_fit,
                validation,
                config.epochs,
                config,
                config.seed + 1000 * fold + 100 * domain_index,
            )
            validation_score, _ = spectral.evaluate_ids(
                screening_model,
                bank,
                validation[nonzero[validation]],
                mode,
                screening_model.context_mean,
                screening_model.context_std,
                config.validation_batch_size,
                False,
            )
            del screening_model

            final_model, _, _ = spectral.train_epochs(
                bank,
                mode,
                outer_fit,
                None,
                best_epoch,
                config,
                config.seed + 10_000 + 1000 * fold + 100 * domain_index,
            )
            held_ids = outer_query[nonzero[outer_query]]
            held_score, _ = spectral.evaluate_ids(
                final_model,
                bank,
                held_ids,
                mode,
                final_model.context_mean,
                final_model.context_std,
                config.validation_batch_size,
                False,
            )
            _save_checkpoint(
                checkpoint_path,
                final_model,
                component,
                domain,
                mode,
                fold,
                best_epoch,
                config,
            )
            fold_reports.append(
                {
                    "status": "trained",
                    "fold": int(fold),
                    "domain": domain,
                    "fit_count": int(len(outer_fit)),
                    "validation_count": int(len(validation)),
                    "held_count": int(len(held_ids)),
                    "best_epoch": int(best_epoch),
                    "validation_cosine": float(validation_score),
                    "held_cosine": float(held_score),
                    "checkpoint": portable_path(checkpoint_path),
                    "history": history,
                }
            )
            del final_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        result[domain] = {"folds": fold_reports}
        del bank
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def train_spectral_calibrators(
    paths: RuntimePaths,
    pipeline_config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    domains: tuple[str, ...] = DOMAINS,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    components: tuple[str, ...] = COMPONENTS,
    force: bool = False,
) -> dict[str, Any]:
    started = time.time()
    _, _, preparation = prepare_training_inputs(
        paths, pipeline_config, device, batch_size, force=False
    )
    results: dict[str, Any] = {}
    for component in components:
        config = BASE_TRAIN_CONFIG if component == "spectral" else JOINT_TRAIN_CONFIG
        results[component] = {
            "config": asdict(config),
            "domains": _train_component(
                paths,
                component,
                domains,
                folds,
                device,
                config,
                force,
            ),
        }
    report = {
        "stage": "train_spectral_calibrators",
        "status": "complete",
        "preparation": preparation,
        "components": results,
        "wall_seconds": float(time.time() - started),
    }
    atomic_json(paths.report_dir / "train_spectral_calibrators.json", report)
    return report
