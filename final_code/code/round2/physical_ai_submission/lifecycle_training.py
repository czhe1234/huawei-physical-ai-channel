"""Build strict OOF lifecycle targets and select the production lifecycle policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .base_field import GEODESIC_NAMES
from .base_runtime import propagation_lifecycle as lifecycle
from .config import PipelineConfig, RuntimePaths
from .data import load_official_data
from .geometry import build_geometry_regime
from .io_utils import atomic_json, portable_path
from .spectral_assets import (
    oof_final_projection_path,
    oof_reference_projection_path,
)
from .training import prepare_training_inputs
from .training_common import FoldMetadata, aggregate_metrics, load_fold_metadata
from .training_common import MetricArrays, official_metric_arrays_torch


TINY = 1e-30


@dataclass(frozen=True)
class LifecycleTrainingConfig:
    batch_size: int = 2
    seed: int = 20_260_906
    required_positive_folds: int = 4
    minimum_group_training_rows: int = 8


def _root(paths: RuntimePaths) -> Path:
    return paths.work_dir / "training" / "lifecycle"


def _support_path(paths: RuntimePaths, name: str) -> Path:
    return _root(paths) / "supports" / f"{name}.npy"


def lifecycle_metrics_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "lifecycle_metrics.npz"


def oof_lifecycle_field_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "oof_lifecycle_field.npy"


def lifecycle_policy_path(paths: RuntimePaths) -> Path:
    return paths.model_dir / "base_field" / "lifecycle_policy.json"


def _write_support(path: Path, value: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float16, shape=value.shape
    )
    output[:] = lifecycle.normalize_support(value).astype(np.float16)
    output.flush()
    del output
    os.replace(partial, path)
    return path


def build_oof_lifecycle_targets(
    paths: RuntimePaths,
    folds: FoldMetadata,
    device: torch.device,
    config: LifecycleTrainingConfig,
    force: bool,
) -> dict[str, Any]:
    target_names = tuple(lifecycle.TARGET_PATHS)
    target_paths = {name: _support_path(paths, f"target_{name}") for name in target_names}
    regime_path = _support_path(paths, "query_regime")
    reference_current_path = _support_path(paths, "reference_current")
    final_current_path = _support_path(paths, "final_current")
    if (
        all(path.is_file() for path in target_paths.values())
        and regime_path.is_file()
        and reference_current_path.is_file()
        and final_current_path.is_file()
        and not force
    ):
        return {
            "status": "reused",
            "targets": target_paths,
            "regime": regime_path,
            "reference_current": reference_current_path,
            "final_current": final_current_path,
        }

    data = load_official_data(paths.data_dir)
    geometry = build_geometry_regime(paths, force=False)
    reference_current_path.parent.mkdir(parents=True, exist_ok=True)
    if force:
        reference_current_path.unlink(missing_ok=True)
        final_current_path.unlink(missing_ok=True)
    reference_current = lifecycle.channel_ad_support(
        np.load(oof_reference_projection_path(paths), mmap_mode="r"),
        reference_current_path,
        device,
        config.batch_size,
    )
    final_current = lifecycle.channel_ad_support(
        np.load(oof_final_projection_path(paths), mmap_mode="r"),
        final_current_path,
        device,
        config.batch_size,
    )
    truth_support = lifecycle.normalize_support(
        np.load(paths.base_dir / "official_spectra" / "angle_delay.npy", mmap_mode="r")
    )
    current_value = lifecycle.normalize_support(reference_current)
    truth_descriptor = lifecycle.support_descriptor(truth_support)
    current_descriptor = lifecycle.support_descriptor(current_value)
    train_features = np.asarray(geometry["train_features"], dtype=np.float32)
    train_visibility = np.asarray(geometry["train_visibility"], dtype=np.float32)
    model_train = np.concatenate((train_visibility, train_features), axis=1)
    train_route = (
        np.asarray(geometry["train_route_occupancy"], dtype=np.float32).reshape(
            len(data.train_positions), -1
        ),
        np.asarray(geometry["train_panorama"], dtype=np.float32).reshape(
            len(data.train_positions), -1
        ),
    )
    truth_peaks = lifecycle.extract_peaks(truth_support)
    lookup = folds.query_lookup
    outputs = {
        name: np.zeros((len(folds.query_ids), 16, 8, 192), dtype=np.float32)
        for name in target_names
    }
    query_regime = np.zeros(len(folds.query_ids), dtype=np.int8)
    fold_reports = []

    for fold in range(5):
        query_ids = folds.query_indices[fold]
        rows = lookup[query_ids]
        train_risk, held_risk, train_regime, held_regime, regime_report = (
            lifecycle.physical_risk_and_regime(
                train_features,
                train_features[query_ids],
                data.train_sites,
                data.train_sites[query_ids],
                folds.fit_indices[fold],
                GEODESIC_NAMES,
            )
        )
        template, confidence, template_report = lifecycle.build_regime_template(
            truth_support,
            truth_descriptor,
            current_value[rows],
            current_descriptor[rows],
            model_train,
            model_train[query_ids],
            data.train_sites,
            data.train_sites[query_ids],
            train_regime,
            held_regime,
            folds.fit_indices[fold],
            config.seed + fold * 101,
        )
        current_peaks = lifecycle.extract_peaks(current_value[rows])
        template_peaks = lifecycle.extract_peaks(template)
        query_route = (
            train_route[0][query_ids],
            train_route[1][query_ids],
        )
        plain, visibility, shifted, target_report = lifecycle.build_lifecycle_targets(
            current_value[rows],
            template,
            data.train_positions[query_ids],
            data.train_sites[query_ids],
            query_ids,
            folds.fit_indices[fold],
            data.train_positions,
            data.train_sites,
            truth_peaks,
            current_peaks,
            template_peaks,
            train_regime,
            held_regime,
            train_risk,
            held_risk,
            train_features,
            train_features[query_ids],
            train_route,
            query_route,
        )
        outputs["template"][rows] = template
        outputs["lifecycle_plain"][rows] = plain
        outputs["lifecycle_visibility"][rows] = visibility
        outputs["lifecycle_shift"][rows] = shifted
        query_regime[rows] = held_regime
        fold_reports.append(
            {
                "fold": fold,
                "fit_count": int(len(folds.fit_indices[fold])),
                "query_count": int(len(query_ids)),
                "confidence_mean": float(np.mean(confidence)),
                "regime": regime_report,
                "template": template_report,
                "targets": target_report,
            }
        )
        print(f"lifecycle targets fold={fold} complete", flush=True)

    for name, value in outputs.items():
        _write_support(target_paths[name], value)
    regime_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = regime_path.with_suffix(".tmp.npy")
    np.save(temporary, query_regime)
    os.replace(temporary, regime_path)
    return {
        "status": "complete",
        "targets": target_paths,
        "regime": regime_path,
        "reference_current": reference_current_path,
        "final_current": final_current_path,
        "folds": fold_reports,
    }


def _statistics(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
    rows: np.ndarray | None = None,
) -> dict[str, float | int]:
    selected = np.arange(len(pas), dtype=np.int64) if rows is None else rows
    return aggregate_metrics(
        MetricArrays(pas, pdp, error, energy),
        np.isin(np.arange(len(pas)), selected),
    )


def generate_lifecycle_metrics(
    paths: RuntimePaths,
    folds: FoldMetadata,
    target_report: dict[str, Any],
    device: torch.device,
    config: LifecycleTrainingConfig,
    force: bool,
) -> dict[str, np.ndarray]:
    output = lifecycle_metrics_path(paths)
    expected_names = ["identity"] + [candidate.name for candidate in lifecycle.candidate_list()]
    if output.is_file() and not force:
        with np.load(output, allow_pickle=False) as loaded:
            if loaded["candidate_names"].astype(str).tolist() == expected_names:
                return {name: np.array(loaded[name], copy=True) for name in loaded.files}
    base = np.load(oof_final_projection_path(paths), mmap_mode="r")
    current = np.load(target_report["final_current"], mmap_mode="r")
    targets = {
        name: np.load(path, mmap_mode="r")
        for name, path in target_report["targets"].items()
    }
    regimes = np.load(target_report["regime"]).astype(np.int8)
    data = load_official_data(paths.data_dir)
    site = data.train_sites[folds.query_ids].astype(np.int8)
    truth_all = data.train_channel(mmap_mode="r")
    candidates = lifecycle.candidate_list()
    names = ["identity"] + [candidate.name for candidate in candidates]
    result = {
        key: np.empty((len(names), len(folds.query_ids)), dtype=np.float64)
        for key in ("pas", "pdp", "error", "energy")
    }
    for start in range(0, len(folds.query_ids), config.batch_size):
        end = min(start + config.batch_size, len(folds.query_ids))
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(
                truth_all[folds.query_ids[start:end]], dtype=np.complex64, copy=True
            ),
            device=device,
        )
        current_tensor = torch.as_tensor(
            np.asarray(current[start:end], dtype=np.float32), device=device
        )
        with torch.inference_mode():
            identity = official_metric_arrays_torch(base_tensor, truth)
            for key, value in zip(
                ("pas", "pdp", "error", "energy"),
                (identity.pas, identity.pdp, identity.squared_error, identity.truth_energy),
            ):
                result[key][0, start:end] = value
            for index, candidate in enumerate(candidates, start=1):
                target = torch.as_tensor(
                    np.asarray(targets[candidate.target][start:end], dtype=np.float32),
                    device=device,
                )
                prediction = lifecycle.apply_candidate(
                    base_tensor.clone(),
                    current_tensor,
                    target,
                    regimes[start:end],
                    site[start:end],
                    candidate,
                )
                values = official_metric_arrays_torch(prediction, truth)
                for key, value in zip(
                    ("pas", "pdp", "error", "energy"),
                    (values.pas, values.pdp, values.squared_error, values.truth_energy),
                ):
                    result[key][index, start:end] = value
        print(f"lifecycle metrics {end}/{len(folds.query_ids)}", flush=True)
    result["candidate_names"] = np.asarray(names, dtype="U72")
    result["query_ids"] = folds.query_ids.astype(np.int32)
    result["fold_id"] = folds.fold_id
    result["site"] = site
    result["regime"] = regimes
    lifecycle_metrics_path(paths).parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **result)
    os.replace(temporary, output)
    return result


def _selected_statistics(
    metrics: dict[str, np.ndarray], selected: np.ndarray, rows: np.ndarray | None = None
) -> dict[str, float | int]:
    point = np.arange(len(selected))
    pas = metrics["pas"][selected, point]
    pdp = metrics["pdp"][selected, point]
    error = metrics["error"][selected, point]
    energy = metrics["energy"][selected, point]
    return _statistics(pas, pdp, error, energy, rows)


def select_lifecycle_policy(
    metrics: dict[str, np.ndarray], config: LifecycleTrainingConfig
) -> dict[str, Any]:
    names = metrics["candidate_names"].astype(str)
    fold_id = metrics["fold_id"].astype(np.int8)
    site = metrics["site"].astype(np.int8)
    regime = metrics["regime"].astype(np.int8)
    selected = np.zeros(len(fold_id), dtype=np.int32)
    decisions = []
    all_candidates = np.arange(len(names), dtype=np.int64)
    for held_fold in range(5):
        for site_value in (0, 1):
            for regime_value in range(4):
                held = np.flatnonzero(
                    (fold_id == held_fold)
                    & (site == site_value)
                    & (regime == regime_value)
                )
                if not len(held):
                    continue
                train = np.flatnonzero(
                    (fold_id != held_fold)
                    & (site == site_value)
                    & (regime == regime_value)
                )
                if len(train) < config.minimum_group_training_rows:
                    choice = 0
                else:
                    choice = max(
                        all_candidates,
                        key=lambda index: float(
                            _statistics(
                                metrics["pas"][index],
                                metrics["pdp"][index],
                                metrics["error"][index],
                                metrics["energy"][index],
                                train,
                            )["score"]
                        ),
                    )
                selected[held] = int(choice)
                decisions.append(
                    {
                        "held_fold": held_fold,
                        "site": site_value,
                        "regime": lifecycle.REGIME_NAMES[regime_value],
                        "regime_index": regime_value,
                        "candidate": str(names[choice]),
                        "candidate_index": int(choice),
                        "train_count": int(len(train)),
                        "held_count": int(len(held)),
                    }
                )
    baseline_selected = np.zeros(len(selected), dtype=np.int32)
    baseline = _selected_statistics(metrics, baseline_selected)
    nested = _selected_statistics(metrics, selected)
    nested_folds = []
    for fold in range(5):
        rows = np.flatnonzero(fold_id == fold)
        value = _selected_statistics(metrics, selected, rows)
        base = _selected_statistics(metrics, baseline_selected, rows)
        nested_folds.append(
            {"fold": fold, **value, "delta_score": float(value["score"] - base["score"])}
        )
    positive_folds = int(sum(row["delta_score"] > 0.0 for row in nested_folds))
    deployment_enabled = (
        positive_folds >= config.required_positive_folds
        and float(nested["score"]) > float(baseline["score"])
    )
    if not deployment_enabled:
        selected[:] = 0
        nested = baseline
        nested_folds = []
        for fold in range(5):
            rows = np.flatnonzero(fold_id == fold)
            base = _selected_statistics(metrics, baseline_selected, rows)
            nested_folds.append({"fold": fold, **base, "delta_score": 0.0})

    final_policy = []
    for site_value in (0, 1):
        for regime_value in range(4):
            rows = np.flatnonzero((site == site_value) & (regime == regime_value))
            choice = 0
            if deployment_enabled and len(rows):
                eligible = [0]
                for candidate_index in range(1, len(names)):
                    deltas = []
                    for fold in range(5):
                        fold_rows = rows[fold_id[rows] == fold]
                        if not len(fold_rows):
                            continue
                        candidate_score = float(
                            _statistics(
                                metrics["pas"][candidate_index],
                                metrics["pdp"][candidate_index],
                                metrics["error"][candidate_index],
                                metrics["energy"][candidate_index],
                                fold_rows,
                            )["score"]
                        )
                        base_score = float(
                            _statistics(
                                metrics["pas"][0],
                                metrics["pdp"][0],
                                metrics["error"][0],
                                metrics["energy"][0],
                                fold_rows,
                            )["score"]
                        )
                        deltas.append(candidate_score - base_score)
                    if deltas and int(np.sum(np.asarray(deltas) > 0.0)) >= min(3, len(deltas)):
                        eligible.append(candidate_index)
                choice = max(
                    eligible,
                    key=lambda index: float(
                        _statistics(
                            metrics["pas"][index],
                            metrics["pdp"][index],
                            metrics["error"][index],
                            metrics["energy"][index],
                            rows,
                        )["score"]
                    ),
                )
            score = (
                float(
                    _statistics(
                        metrics["pas"][choice],
                        metrics["pdp"][choice],
                        metrics["error"][choice],
                        metrics["energy"][choice],
                        rows,
                    )["score"]
                )
                if len(rows)
                else None
            )
            final_policy.append(
                {
                    "site": site_value,
                    "regime": lifecycle.REGIME_NAMES[regime_value],
                    "regime_index": regime_value,
                    "candidate": str(names[choice]),
                    "candidate_index": int(choice),
                    "count": int(len(rows)),
                    "oof_score": score,
                }
            )
    return {
        "baseline": baseline,
        "nested": nested,
        "nested_folds": nested_folds,
        "nested_decisions": decisions,
        "positive_folds": positive_folds,
        "deployment_enabled": deployment_enabled,
        "final_policy": final_policy,
        "selected_candidate": selected,
    }


def write_oof_lifecycle_field(
    paths: RuntimePaths,
    folds: FoldMetadata,
    target_report: dict[str, Any],
    selection: dict[str, Any],
    device: torch.device,
    config: LifecycleTrainingConfig,
    force: bool,
) -> Path:
    output = oof_lifecycle_field_path(paths)
    selected = np.asarray(selection["selected_candidate"], dtype=np.int64)
    if output.is_file() and not force:
        value = np.load(output, mmap_mode="r")
        if value.shape == (len(selected), 256, 4, 192) and value.dtype == np.complex64:
            return output
    base = np.load(oof_final_projection_path(paths), mmap_mode="r")
    current = np.load(target_report["final_current"], mmap_mode="r")
    targets = {
        name: np.load(path, mmap_mode="r")
        for name, path in target_report["targets"].items()
    }
    regimes = np.load(target_report["regime"]).astype(np.int8)
    data = load_official_data(paths.data_dir)
    site = data.train_sites[folds.query_ids]
    names = ["identity"] + [candidate.name for candidate in lifecycle.candidate_list()]
    candidate_by_name = {candidate.name: candidate for candidate in lifecycle.candidate_list()}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    target_output = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.complex64,
        shape=(len(selected), 256, 4, 192),
    )
    for start in range(0, len(selected), config.batch_size):
        end = min(start + config.batch_size, len(selected))
        block = np.array(base[start:end], dtype=np.complex64, copy=True)
        local_selected = selected[start:end]
        for candidate_index in np.unique(local_selected):
            rows = np.flatnonzero(local_selected == candidate_index)
            name = names[int(candidate_index)]
            if name == "identity":
                continue
            candidate = candidate_by_name[name]
            with torch.inference_mode():
                prediction = lifecycle.apply_candidate(
                    torch.as_tensor(block[rows], device=device),
                    torch.as_tensor(
                        np.asarray(current[start:end][rows], dtype=np.float32), device=device
                    ),
                    torch.as_tensor(
                        np.asarray(targets[candidate.target][start:end][rows], dtype=np.float32),
                        device=device,
                    ),
                    regimes[start:end][rows],
                    site[start:end][rows],
                    candidate,
                )
            block[rows] = prediction.cpu().numpy().astype(np.complex64)
        target_output[start:end] = block
        target_output.flush()
    del target_output
    os.replace(partial, output)
    return output


def train_lifecycle_policy(
    paths: RuntimePaths,
    pipeline_config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool = False,
    training_config: LifecycleTrainingConfig | None = None,
) -> dict[str, Any]:
    """Construct OOF lifecycle supervision, select policy, and write OOF field."""

    started = time.time()
    config = training_config or LifecycleTrainingConfig(
        batch_size=max(1, min(batch_size, 4))
    )
    _, layout, preparation = prepare_training_inputs(
        paths, pipeline_config, device, batch_size, force=False
    )
    folds = load_fold_metadata(layout.fold_cache, layout.fold_metadata)
    targets = build_oof_lifecycle_targets(paths, folds, device, config, force)
    metrics = generate_lifecycle_metrics(
        paths, folds, targets, device, config, force
    )
    selection = select_lifecycle_policy(metrics, config)
    field = write_oof_lifecycle_field(
        paths, folds, targets, selection, device, config, force
    )
    winner = {
        key: value for key, value in selection.items() if key != "selected_candidate"
    }
    payload = {
        "experiment": "Round2-V17-official-retraining",
        "baseline": selection["baseline"],
        "winner": winner,
    }
    atomic_json(lifecycle_policy_path(paths), payload)
    report = {
        "stage": "train_lifecycle_policy",
        "status": "complete",
        "config": asdict(config),
        "preparation": preparation,
        "targets": {
            "status": targets["status"],
            "targets": {
                name: portable_path(path) for name, path in targets["targets"].items()
            },
            "regime": portable_path(targets["regime"]),
            "reference_current": portable_path(targets["reference_current"]),
            "final_current": portable_path(targets["final_current"]),
            "folds": targets.get("folds", []),
        },
        "metrics": portable_path(lifecycle_metrics_path(paths)),
        "selection": winner,
        "field": portable_path(field),
        "policy": portable_path(lifecycle_policy_path(paths)),
        "wall_seconds": float(time.time() - started),
        "device": str(device),
        "peak_torch_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
        "measured_bottleneck": "OOF lifecycle target synthesis and 37-candidate projection",
    }
    atomic_json(paths.report_dir / "train_lifecycle_policy.json", report)
    return report
