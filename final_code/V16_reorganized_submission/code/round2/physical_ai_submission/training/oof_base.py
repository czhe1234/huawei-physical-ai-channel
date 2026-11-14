from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from ..ai_enhancement import joint_spectral_model as joint
from ..ai_enhancement import spectral_calibrator as spectral
from ..ai_enhancement.joint_spectral_projection import Candidate as JointCandidate
from ..ai_enhancement.joint_spectral_projection import (
    apply_candidate as apply_joint_candidate,
)
from ..config import PipelineConfig, RuntimePaths
from ..data.artifact_io import atomic_json, portable_path
from .lifecycle import oof_lifecycle_field_path
from .spectral_models import prepare_training_inputs
from .common import FoldMetadata, load_fold_metadata


TINY = 1e-30


def _root(paths: RuntimePaths) -> Path:
    return paths.work_dir / "training" / "oof_base"


def oof_endpoint_path(paths: RuntimePaths, component: str, domain: str) -> Path:
    return _root(paths) / "endpoints" / f"{component }_{domain }.npy"


def oof_reliability_path(paths: RuntimePaths, domain: str) -> Path:
    return _root(paths) / "endpoints" / f"{domain }_reliability.npy"


def oof_joint_field_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "oof_joint_spectral_field.npy"


def _checkpoint_path(
    paths: RuntimePaths, component: str, domain: str, fold: int
) -> Path:
    return paths.model_dir / "base_field" / component / f"{domain }_fold_{fold }.pt"


def _instantiate(
    checkpoint: dict[str, Any],
    bank: Any,
    mode: str,
    device: torch.device,
) -> tuple[spectral.SpectralCalibrator, torch.Tensor, torch.Tensor]:
    config = spectral.TrainConfig(**checkpoint["config"])
    context_mean = checkpoint["context_mean"].to(device=device, dtype=torch.float32)
    context_std = checkpoint["context_std"].to(device=device, dtype=torch.float32)
    point = torch.zeros(1, dtype=torch.long, device=device)
    slice_id = torch.zeros(1, dtype=torch.long, device=device)
    feature, context, _, _ = bank.sample(
        point, slice_id, mode, context_mean, context_std
    )
    model = spectral.SpectralCalibrator(
        bank.domain, feature.shape[1], context.shape[1], config
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, context_mean, context_std


def _write_array(path: Path, value: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float16, shape=value.shape
    )
    output[:] = value.astype(np.float16)
    output.flush()
    del output
    os.replace(partial, path)
    return path


def _predict_component(
    paths: RuntimePaths,
    component: str,
    domain: str,
    folds: FoldMetadata,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> tuple[Path, list[dict[str, Any]]]:
    output = oof_endpoint_path(paths, component, domain)
    expected_shape = (
        (len(folds.query_ids), 2, 16, 8, 4, 192)
        if domain == "pas"
        else (len(folds.query_ids), 256, 4, 192)
    )
    if output.is_file() and not force:
        cached = np.load(output, mmap_mode="r")
        if cached.shape == expected_shape and cached.dtype == np.float16:
            return output, [{"status": "reused", "output": portable_path(output)}]

    bank = (
        spectral.DomainBank(domain, device)
        if component == "spectral"
        else joint.JointDomainBank(domain, "train", device, force=False)
    )
    mode = "mean" if component == "spectral" else "joint"
    raw_shape = (
        (len(folds.query_ids), 768, 2, 16, 8)
        if domain == "pas"
        else (len(folds.query_ids), 1024, 192)
    )
    combined = np.empty(raw_shape, dtype=np.float16)
    lookup = folds.query_lookup
    reports = []
    for fold in range(5):
        checkpoint_path = _checkpoint_path(paths, component, domain, fold)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model, context_mean, context_std = _instantiate(checkpoint, bank, mode, device)
        query_ids = folds.query_indices[fold]
        score, prediction = spectral.evaluate_ids(
            model,
            bank,
            query_ids,
            mode,
            context_mean,
            context_std,
            batch_size,
            True,
        )
        if prediction is None:
            raise RuntimeError("fold-matched spectral prediction returned no tensor")
        combined[lookup[query_ids]] = prediction
        reports.append(
            {
                "fold": fold,
                "checkpoint": portable_path(checkpoint_path),
                "epoch": int(checkpoint["epoch"]),
                "query_count": int(len(query_ids)),
                "truth_cosine": float(score),
            }
        )
        del model, checkpoint, prediction
        if device.type == "cuda":
            torch.cuda.empty_cache()
    shaped = spectral.reshape_oof_prediction(domain, combined)
    _write_array(output, shaped)
    del combined, shaped, bank
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output, reports


def build_oof_spectral_endpoints(
    paths: RuntimePaths,
    pipeline_config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool = False,
) -> dict[str, Any]:
    started = time.time()
    _, layout, preparation = prepare_training_inputs(
        paths, pipeline_config, device, batch_size, force=False
    )
    folds = load_fold_metadata(layout.fold_cache, layout.fold_metadata)
    outputs: dict[str, Path] = {}
    reports: dict[str, Any] = {}
    for component in ("spectral", "joint"):
        for domain in ("pas", "pdp"):
            output, local = _predict_component(
                paths,
                component,
                domain,
                folds,
                device,
                batch_size,
                force,
            )
            outputs[f"{component }_{domain }"] = output
            reports[f"{component }_{domain }"] = local

    for domain in ("pas", "pdp"):
        output = oof_reliability_path(paths, domain)
        if not output.is_file() or force:
            reliability = joint.spectral_reliability(domain, "train", force=False)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".tmp.npy")
            np.save(
                temporary,
                np.asarray(reliability[folds.query_ids], dtype=np.float32),
            )
            os.replace(temporary, output)
        outputs[f"{domain }_reliability"] = output
    report = {
        "stage": "build_oof_spectral_endpoints",
        "status": "complete",
        "preparation": preparation,
        "outputs": {name: portable_path(path) for name, path in outputs.items()},
        "folds": reports,
        "wall_seconds": float(time.time() - started),
        "device": str(device),
        "peak_torch_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
        "measured_bottleneck": "fold-matched spectral checkpoint inference",
    }
    atomic_json(paths.report_dir / "build_oof_spectral_endpoints.json", report)
    return report


def build_oof_joint_field(
    paths: RuntimePaths,
    device: torch.device,
    batch_size: int,
    force: bool = False,
) -> dict[str, Any]:
    started = time.time()
    output = oof_joint_field_path(paths)
    base = np.load(oof_lifecycle_field_path(paths), mmap_mode="r")
    if output.is_file() and not force:
        value = np.load(output, mmap_mode="r")
        if value.shape == base.shape and value.dtype == np.complex64:
            return {
                "stage": "build_oof_joint_field",
                "status": "reused",
                "output": portable_path(output),
            }
    old_pas = np.load(oof_endpoint_path(paths, "spectral", "pas"), mmap_mode="r")
    old_pdp = np.load(oof_endpoint_path(paths, "spectral", "pdp"), mmap_mode="r")
    joint_pas = np.load(oof_endpoint_path(paths, "joint", "pas"), mmap_mode="r")
    joint_pdp = np.load(oof_endpoint_path(paths, "joint", "pdp"), mmap_mode="r")
    policy = JointCandidate(
        name="frozen_joint_endpoint",
        pas_beta=0.75,
        pdp_beta=0.50,
        alpha_pas=0.45,
        alpha_pdp=0.35,
        order="pas_pdp",
        cycles=4,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    target = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        with torch.inference_mode():
            prediction = apply_joint_candidate(
                torch.as_tensor(
                    np.array(base[start:end], dtype=np.complex64, copy=True),
                    device=device,
                ),
                torch.as_tensor(
                    np.asarray(old_pas[start:end], dtype=np.float32), device=device
                ),
                torch.as_tensor(
                    np.asarray(joint_pas[start:end], dtype=np.float32), device=device
                ),
                torch.as_tensor(
                    np.asarray(old_pdp[start:end], dtype=np.float32), device=device
                ),
                torch.as_tensor(
                    np.asarray(joint_pdp[start:end], dtype=np.float32), device=device
                ),
                policy,
            )
        target[start:end] = prediction.cpu().numpy().astype(np.complex64)
        target.flush()
        print(f"OOF joint field {end }/{len (base )}", flush=True)
    del target
    os.replace(partial, output)
    report = {
        "stage": "build_oof_joint_field",
        "status": "complete",
        "output": portable_path(output),
        "wall_seconds": float(time.time() - started),
        "device": str(device),
        "peak_torch_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
        "measured_bottleneck": "batched four-cycle joint spectral projection",
    }
    atomic_json(paths.report_dir / "build_oof_joint_field.json", report)
    return report
