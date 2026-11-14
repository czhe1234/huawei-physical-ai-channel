from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from ..synthesis.spectral_projection import pdp_distribution
from ..ai_enhancement.link_gate import (
    LinkGate,
    cosine_by_link,
    link_coordinates,
    project,
    spectral_features,
)
from ..ai_enhancement.link_gate_model import predict_alpha
from ..config import PipelineConfig, RuntimePaths
from ..data.official_data import load_official_data
from ..scene_prior.component_features import load_model_inputs
from ..geometry_context.map_features import build_geometry_regime
from ..data.artifact_io import atomic_json, atomic_npz, portable_path
from .oof_base import oof_joint_field_path, oof_reliability_path
from ..scene_prior.component_aggregation import cache_dir as component_token_dir
from .spectral_models import prepare_training_inputs
from .common import atomic_torch_save, load_fold_metadata, set_seed


TINY = 1e-30
FREQUENCIES = np.arange(-96, 96, dtype=np.float32) * 30_000.0


@dataclass(frozen=True)
class LinkTrainingConfig:
    batch_size: int = 8
    epochs: int = 36
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    gate_penalty: float = 2e-4
    hidden: int = 320
    dropout: float = 0.10
    alpha_scale: float = 0.5
    order: str = "pas_pdp"
    cycles: int = 2
    seed: int = 2_026_080_666


def _root(paths: RuntimePaths) -> Path:
    return paths.work_dir / "training" / "link_gate"


def _base_pdp_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "oof_base_pdp.npy"


def _component_pdp_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "oof_component_pdp.npy"


def _context_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "oof_point_context.npy"


def _fold_alpha_path(paths: RuntimePaths, fold: int) -> Path:
    return _root(paths) / "fold_alpha" / f"fold_{fold }.npz"


def oof_alpha_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "oof_alpha.npy"


def oof_component_link_field_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "oof_component_link_field.npy"


def link_checkpoint_path(paths: RuntimePaths, fold: int) -> Path:
    return paths.model_dir / "base_field" / "component_link" / f"fold_{fold }.pt"


def _write_float16(path: Path, shape: tuple[int, ...]) -> tuple[np.memmap, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    return (
        np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=shape),
        partial,
    )


def build_oof_base_pdp(
    paths: RuntimePaths,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    output = _base_pdp_path(paths)
    base = np.load(oof_joint_field_path(paths), mmap_mode="r")
    if output.is_file() and not force:
        cached = np.load(output, mmap_mode="r")
        if cached.shape == base.shape and cached.dtype == np.float16:
            return cached
    target, partial = _write_float16(output, base.shape)
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        channel = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        with torch.inference_mode():
            target[start:end] = (
                pdp_distribution(channel).cpu().numpy().astype(np.float16)
            )
        target.flush()
        print(f"LinkGate base PDP {end }/{len (base )}", flush=True)
    del target
    os.replace(partial, output)
    return np.load(output, mmap_mode="r")


def build_oof_component_pdp(
    paths: RuntimePaths,
    query_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    output = _component_pdp_path(paths)
    expected = (len(query_ids), 256, 4, 192)
    if output.is_file() and not force:
        cached = np.load(output, mmap_mode="r")
        if cached.shape == expected and cached.dtype == np.float16:
            return cached
    token_root = component_token_dir(paths, "train")
    coefficient = np.load(token_root / "coefficient.npy", mmap_mode="r")
    tau = np.load(token_root / "tau.npy", mmap_mode="r")
    mask = np.load(token_root / "mask.npy", mmap_mode="r")
    target, partial = _write_float16(output, expected)
    frequency = torch.as_tensor(FREQUENCIES, dtype=torch.float32, device=device)
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        ids = query_ids[start:end]
        coefficient_tensor = torch.as_tensor(
            np.array(coefficient[ids], dtype=np.complex64, copy=True), device=device
        )
        tau_tensor = torch.as_tensor(
            np.array(tau[ids], dtype=np.float32, copy=True), device=device
        )
        mask_tensor = torch.as_tensor(np.asarray(mask[ids]), device=device)
        phase = torch.exp(
            -2j * torch.pi * tau_tensor[:, :, None] * frequency[None, None]
        )
        channel = torch.einsum(
            "bk,bkmu,bks->bmus",
            mask_tensor.to(coefficient_tensor.dtype),
            coefficient_tensor,
            phase,
        )
        with torch.inference_mode():
            target[start:end] = (
                pdp_distribution(channel).cpu().numpy().astype(np.float16)
            )
        target.flush()
        print(f"LinkGate component PDP {end }/{len (query_ids )}", flush=True)
    del target
    os.replace(partial, output)
    return np.load(output, mmap_mode="r")


def build_oof_point_context(
    paths: RuntimePaths,
    query_ids: np.ndarray,
    force: bool,
) -> np.ndarray:
    output = _context_path(paths)
    if output.is_file() and not force:
        cached = np.load(output, mmap_mode="r")
        if cached.shape[0] == len(query_ids):
            return cached
    data = load_official_data(paths.data_dir)
    geometry = build_geometry_regime(paths, force=False)
    features, _ = load_model_inputs(paths, "train")
    mask = np.asarray(
        np.load(component_token_dir(paths, "train") / "mask.npy", mmap_mode="r"),
        dtype=bool,
    )
    token_mean = np.empty((len(query_ids), features.shape[2]), dtype=np.float32)
    for start in range(0, len(query_ids), 64):
        end = min(start + 64, len(query_ids))
        ids = query_ids[start:end]
        valid = mask[ids, :, None].astype(np.float32)
        token_mean[start:end] = (
            np.asarray(features[ids], dtype=np.float32) * valid
        ).sum(axis=1) / valid.sum(axis=1).clip(min=1.0)
    train_xy = data.train_positions[:, :2]
    query_xy = train_xy[query_ids]
    center = train_xy.mean(axis=0)
    scale = train_xy.std(axis=0) + 1e-6
    train_relative = train_xy - data.setup.bs_positions[data.train_sites, :2]
    query_sites = data.train_sites[query_ids]
    query_relative = query_xy - data.setup.bs_positions[query_sites, :2]
    relative_scale = train_relative.std(axis=0) + 1e-6
    bearing = np.arctan2(query_relative[:, 1], query_relative[:, 0])
    regime = np.asarray(geometry["train_regime"], dtype=np.int64)[query_ids]
    reliability = np.asarray(
        np.load(oof_reliability_path(paths, "pdp"), mmap_mode="r"),
        dtype=np.float32,
    )
    extra = np.concatenate(
        (
            ((query_xy - center) / scale).astype(np.float32),
            (query_relative / relative_scale).astype(np.float32),
            np.sin(bearing)[:, None].astype(np.float32),
            np.cos(bearing)[:, None].astype(np.float32),
            np.eye(2, dtype=np.float32)[query_sites],
            np.eye(4, dtype=np.float32)[regime],
            reliability,
        ),
        axis=1,
    )
    context = np.concatenate((token_mean, extra), axis=1).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npy")
    np.save(temporary, context)
    os.replace(temporary, output)
    return np.load(output, mmap_mode="r")


def _train_fold(
    paths: RuntimePaths,
    fold: int,
    fold_id: np.ndarray,
    nonzero: np.ndarray,
    base_pdp: np.ndarray,
    component_pdp: np.ndarray,
    truth_pdp: np.ndarray,
    query_ids: np.ndarray,
    context: np.ndarray,
    config: LinkTrainingConfig,
    device: torch.device,
    force: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    checkpoint_path = link_checkpoint_path(paths, fold)
    alpha_path = _fold_alpha_path(paths, fold)
    held_points = np.flatnonzero((fold_id == fold) & nonzero)
    if checkpoint_path.is_file() and alpha_path.is_file() and not force:
        with np.load(alpha_path, allow_pickle=False) as loaded:
            if np.array_equal(loaded["rows"], held_points):
                return loaded["alpha"].astype(np.float32), {
                    "fold": fold,
                    "status": "reused",
                    "checkpoint": portable_path(checkpoint_path),
                    "held_count": int(len(held_points)),
                }
    set_seed(config.seed + fold * 1000)
    train_points = np.flatnonzero((fold_id != fold) & nonzero)
    context_mean = context[train_points].mean(axis=0).astype(np.float32)
    context_std = context[train_points].std(axis=0).astype(np.float32)
    context_std[context_std < 1e-5] = 1.0
    link = link_coordinates()
    input_dim = 24 * 4 + 3 + context.shape[1] + link.shape[1]
    model = LinkGate(input_dim, config.hidden, config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1), eta_min=config.learning_rate * 0.1
    )
    link_tensor = torch.as_tensor(link, device=device)
    rng = np.random.default_rng(config.seed + fold * 1000)
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(train_points)
        cosine_rows = []
        alpha_rows = []
        for start in range(0, len(order), config.batch_size):
            ids = order[start : start + config.batch_size]
            base = torch.as_tensor(
                np.asarray(base_pdp[ids], dtype=np.float32).reshape(
                    len(ids), 1024, 192
                ),
                device=device,
            )
            raw = torch.as_tensor(
                np.asarray(component_pdp[ids], dtype=np.float32).reshape(len(ids), 1024, 192),
                device=device,
            )
            truth = torch.as_tensor(
                np.asarray(truth_pdp[query_ids[ids]], dtype=np.float32).reshape(
                    len(ids), 1024, 192
                ),
                device=device,
            )
            local_context = torch.as_tensor(
                (np.asarray(context[ids]) - context_mean) / context_std,
                device=device,
            )
            features = spectral_features(base, raw, local_context, link_tensor)
            alpha = model(features)
            desired = (1.0 - alpha[..., None]) * base + alpha[..., None] * raw
            desired = desired / desired.sum(dim=2, keepdim=True).clamp_min(TINY)
            cosine = cosine_by_link(desired, truth)
            loss = -cosine.mean() + config.gate_penalty * alpha.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            cosine_rows.append(float(cosine.mean().detach().cpu()))
            alpha_rows.append(float(alpha.mean().detach().cpu()))
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_pdp_cosine": float(np.mean(cosine_rows)),
                "train_alpha_mean": float(np.mean(alpha_rows)),
            }
        )
        if epoch % 6 == 0 or epoch == config.epochs:
            print(
                f"LinkGate fold={fold } epoch={epoch }/{config .epochs } "
                f"cosine={history [-1 ]['train_pdp_cosine']:.6f} "
                f"alpha={history [-1 ]['train_alpha_mean']:.4f}",
                flush=True,
            )
    prediction = predict_alpha(
        model,
        held_points,
        base_pdp,
        component_pdp,
        context,
        context_mean,
        context_std,
        link_tensor,
        device,
        config.batch_size,
    )
    checkpoint = {
        "state_dict": model.state_dict(),
        "context_mean": context_mean,
        "context_std": context_std,
        "input_dim": input_dim,
        "config": asdict(config),
    }
    atomic_torch_save(checkpoint_path, checkpoint)
    atomic_npz(alpha_path, rows=held_points.astype(np.int32), alpha=prediction)
    return prediction, {
        "fold": fold,
        "status": "trained",
        "train_count": int(len(train_points)),
        "held_count": int(len(held_points)),
        "checkpoint": portable_path(checkpoint_path),
        "history": history,
    }


def _write_oof_component_link_field(
    paths: RuntimePaths,
    base_pdp: np.ndarray,
    component_pdp: np.ndarray,
    alpha: np.ndarray,
    config: LinkTrainingConfig,
    device: torch.device,
    force: bool,
) -> Path:
    output = oof_component_link_field_path(paths)
    base = np.load(oof_joint_field_path(paths), mmap_mode="r")
    if output.is_file() and not force:
        cached = np.load(output, mmap_mode="r")
        if cached.shape == base.shape and cached.dtype == np.complex64:
            return output
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    target = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), config.batch_size):
        end = min(start + config.batch_size, len(base))
        reference = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        local_base = torch.as_tensor(
            np.asarray(base_pdp[start:end], dtype=np.float32), device=device
        )
        local_raw = torch.as_tensor(
            np.asarray(component_pdp[start:end], dtype=np.float32), device=device
        )
        local_alpha = torch.clamp(
            config.alpha_scale
            * torch.as_tensor(alpha[start:end], dtype=torch.float32, device=device),
            0.0,
            1.0,
        ).reshape(end - start, 256, 4)
        desired = (1.0 - local_alpha[..., None]) * local_base + local_alpha[
            ..., None
        ] * local_raw
        desired = desired / desired.sum(dim=3, keepdim=True).clamp_min(TINY)
        with torch.inference_mode():
            prediction = project(reference, desired, config.cycles, config.order)
        target[start:end] = prediction.cpu().numpy().astype(np.complex64)
        target.flush()
    del target
    os.replace(partial, output)
    return output


def train_link_gate_models(
    paths: RuntimePaths,
    pipeline_config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool = False,
    training_config: LinkTrainingConfig | None = None,
) -> dict[str, Any]:
    pass

    started = time.time()
    config = training_config or LinkTrainingConfig()
    data, layout, preparation = prepare_training_inputs(
        paths, pipeline_config, device, batch_size, force=False
    )
    folds = load_fold_metadata(layout.fold_cache, layout.fold_metadata)
    base_pdp = build_oof_base_pdp(paths, device, min(batch_size, 8), force)
    component_pdp = build_oof_component_pdp(
        paths, folds.query_ids, device, min(batch_size, 8), force
    )
    context = build_oof_point_context(paths, folds.query_ids, force)
    truth_pdp = np.load(paths.base_dir / "official_spectra" / "pdp.npy", mmap_mode="r")
    with np.load(
        paths.base_dir / "official_spectra" / "descriptors.npz", allow_pickle=False
    ) as descriptors:
        nonzero = descriptors["valid"].astype(bool)[folds.query_ids]
    alpha = np.zeros((len(folds.query_ids), 1024), dtype=np.float32)
    reports = []
    for fold in range(5):
        held = np.flatnonzero((folds.fold_id == fold) & nonzero)
        prediction, report = _train_fold(
            paths,
            fold,
            folds.fold_id,
            nonzero,
            base_pdp,
            component_pdp,
            truth_pdp,
            folds.query_ids,
            context,
            config,
            device,
            force,
        )
        alpha[held] = prediction
        reports.append(report)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    oof_alpha_path(paths).parent.mkdir(parents=True, exist_ok=True)
    temporary = oof_alpha_path(paths).with_suffix(".tmp.npy")
    np.save(temporary, alpha.astype(np.float16))
    os.replace(temporary, oof_alpha_path(paths))
    field = _write_oof_component_link_field(
        paths, base_pdp, component_pdp, alpha, config, device, force
    )
    report = {
        "stage": "train_link_gate_models",
        "status": "complete",
        "config": asdict(config),
        "preparation": preparation,
        "nonzero_count": int(np.sum(nonzero)),
        "base_pdp": portable_path(_base_pdp_path(paths)),
        "component_pdp": portable_path(_component_pdp_path(paths)),
        "context": portable_path(_context_path(paths)),
        "alpha": portable_path(oof_alpha_path(paths)),
        "field": portable_path(field),
        "folds": reports,
        "wall_seconds": float(time.time() - started),
        "device": str(device),
        "peak_torch_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
        "measured_bottleneck": "five LinkGate fits over 1024 link spectra",
    }
    atomic_json(paths.report_dir / "train_link_gate_models.json", report)
    return report
