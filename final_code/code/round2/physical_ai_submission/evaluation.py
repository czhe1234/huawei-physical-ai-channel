"""Construct and score the canonical strict fixed-500 retrained channel."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .base_runtime import reliable_angle_field as reliability
from .config import PipelineConfig, RuntimePaths
from .data import load_official_data
from .io_utils import atomic_json, portable_path, validate_channel
from .link_training import oof_path_link_field_path
from .oof_base import oof_endpoint_path
from .renderer import project_complete_channel, render_endpoint
from .training import prepare_training_inputs
from .training_common import (
    MetricArrays,
    aggregate_metrics,
    load_fold_metadata,
    official_metric_arrays_torch,
)


TINY = 1e-30


def _root(paths: RuntimePaths) -> Path:
    return paths.work_dir / "training" / "evaluation"


def oof_reliable_field_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "oof_reliable_base_field.npy"


def fixed500_channel_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "fixed500_channel.npy"


def _physical_pas_path(paths: RuntimePaths, configuration: str) -> Path:
    return _root(paths) / f"oof_{configuration}_physical_pas.npy"


def _slice_physical_pas(
    paths: RuntimePaths,
    configuration: str,
    query_ids: np.ndarray,
    force: bool,
) -> Path:
    output = _physical_pas_path(paths, configuration)
    source_path = (
        paths.work_dir
        / "physical_spectra"
        / "train"
        / configuration
        / "pas_mean.npy"
    )
    source = np.load(source_path, mmap_mode="r")
    expected = (len(query_ids), *source.shape[1:])
    if output.is_file() and not force:
        cached = np.load(output, mmap_mode="r")
        if cached.shape == expected and cached.dtype == source.dtype:
            return output
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    target = np.lib.format.open_memmap(
        partial, mode="w+", dtype=source.dtype, shape=expected
    )
    for start in range(0, len(query_ids), 8):
        end = min(start + 8, len(query_ids))
        target[start:end] = source[query_ids[start:end]]
    target.flush()
    del target
    os.replace(partial, output)
    return output


def build_oof_reliable_field(
    paths: RuntimePaths,
    query_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    output = oof_reliable_field_path(paths)
    base = np.load(oof_path_link_field_path(paths), mmap_mode="r")
    if output.is_file() and not force:
        cached = np.load(output, mmap_mode="r")
        if cached.shape == base.shape and cached.dtype == np.complex64:
            return {
                "status": "reused",
                "output": portable_path(output),
            }
    stage = _root(paths) / "reliability"
    stage.mkdir(parents=True, exist_ok=True)
    reliability.PATH_LINK_TEST = oof_path_link_field_path(paths)
    reliability.CALIBRATED_PAS = oof_endpoint_path(paths, "spectral", "pas")
    reliability.JOINT_ENDPOINT_TEST = oof_endpoint_path(paths, "joint", "pas")
    reliability.BASE_PHYSICAL_PAS = _slice_physical_pas(
        paths, "base", query_ids, force
    )
    reliability.SCATTER_PHYSICAL_PAS = _slice_physical_pas(
        paths, "scatter", query_ids, force
    )
    reliability.RAW_FEATURES = stage / "raw_features.npy"
    reliability.GATE_SCORE = stage / "gate_score.npy"
    reliability.GATE = stage / "gate.npy"
    reliability.OUTPUT = output
    if force:
        for path in (
            reliability.RAW_FEATURES,
            reliability.GATE_SCORE,
            reliability.GATE,
            output,
        ):
            path.unlink(missing_ok=True)
    sources = reliability.source_arrays()
    gate = reliability.build_gate(base, sources, device, batch_size, force=False)
    generation = reliability.generate(base, sources, gate, device, batch_size)
    return {
        "status": "complete",
        "output": portable_path(output),
        "generation": generation,
    }


def render_fixed500(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> Path:
    output = fixed500_channel_path(paths)
    base = np.load(oof_reliable_field_path(paths), mmap_mode="r")
    if output.is_file() and not force:
        cached = np.load(output, mmap_mode="r")
        if cached.shape == base.shape and cached.dtype == np.complex64:
            return output
    state_path = paths.work_dir / "training" / "path_state" / "oof_state.npz"
    with np.load(state_path, allow_pickle=False) as state:
        mass = state["mass"].astype(np.float32)
        coordinate = state["coordinate"].astype(np.float32)
        port = state["port"].astype(np.float32)
    if len(mass) != len(base):
        raise RuntimeError("PathState OOF rows do not match the OOF base field")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    target = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        endpoint_pas, endpoint_pdp = render_endpoint(
            mass[start:end], coordinate[start:end], port[start:end], device
        )
        prediction = project_complete_channel(
            base_tensor, endpoint_pas, endpoint_pdp, config
        )
        target[start:end] = prediction.cpu().numpy().astype(np.complex64)
        target.flush()
        print(f"fixed500 render {end}/{len(base)}", flush=True)
    del target
    os.replace(partial, output)
    return output


def _score_fixed500(
    prediction_path: Path,
    truth: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    prediction = np.load(prediction_path, mmap_mode="r")
    arrays = []
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        arrays.append(
            official_metric_arrays_torch(
                torch.as_tensor(
                    np.array(prediction[start:end], dtype=np.complex64, copy=True),
                    device=device,
                ),
                torch.as_tensor(
                    np.array(
                        truth[query_ids[start:end]], dtype=np.complex64, copy=True
                    ),
                    device=device,
                ),
            )
        )
    combined = MetricArrays(
        pas=np.concatenate([value.pas for value in arrays]),
        pdp=np.concatenate([value.pdp for value in arrays]),
        squared_error=np.concatenate([value.squared_error for value in arrays]),
        truth_energy=np.concatenate([value.truth_energy for value in arrays]),
    )
    return {
        "overall": aggregate_metrics(combined),
        "folds": [
            {"fold": fold, **aggregate_metrics(combined, fold_id == fold)}
            for fold in range(5)
        ],
    }


def evaluate_retrained(
    paths: RuntimePaths,
    pipeline_config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool = False,
    success_threshold: float = 0.670000,
) -> dict[str, Any]:
    """Run the final strict fixed500 assembly and official local score."""

    started = time.time()
    data, layout, preparation = prepare_training_inputs(
        paths, pipeline_config, device, batch_size, force=False
    )
    folds = load_fold_metadata(layout.fold_cache, layout.fold_metadata)
    reliable_report = build_oof_reliable_field(
        paths, folds.query_ids, device, min(batch_size, 8), force
    )
    output = render_fixed500(
        paths, pipeline_config, device, min(batch_size, 8), force
    )
    validation = validate_channel(output, len(folds.query_ids))
    metrics = _score_fixed500(
        output,
        data.train_channel(mmap_mode="r"),
        folds.query_ids,
        folds.fold_id,
        device,
        min(batch_size, 8),
    )
    passed = float(metrics["overall"]["score"]) >= success_threshold
    report = {
        "stage": "evaluate_retrained",
        "status": "complete",
        "preparation": preparation,
        "reliability": reliable_report,
        "output": validation,
        "metrics": metrics,
        "success_threshold": success_threshold,
        "quality_gate_passed": passed,
        "v16_local_reference": 0.686375,
        "delta_vs_v16_local": float(metrics["overall"]["score"] - 0.686375),
        "wall_seconds": float(time.time() - started),
        "device": str(device),
        "peak_torch_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
        "measured_bottleneck": "reliability projection, PathState render, and official scoring",
    }
    atomic_json(paths.report_dir / "evaluate_retrained.json", report)
    return report

