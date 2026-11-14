"""Train the five PathState checkpoints directly from official Train channels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import PipelineConfig, RuntimePaths
from .data import load_official_data
from .features import load_model_inputs
from .io_utils import atomic_json, atomic_npz, portable_path
from .model import ModelConfig, PathStateModel, _batch, checkpoint_path
from .path_tokens import cache_dir as path_token_dir
from .training import prepare_training_inputs
from .training_common import atomic_torch_save, load_fold_metadata, set_seed


TINY = 1e-30
PERIODS = np.asarray((16.0, 8.0, 192.0), dtype=np.float32)


@dataclass(frozen=True)
class PathTrainingConfig:
    epochs: int = 24
    minimum_epochs: int = 8
    validation_interval: int = 2
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 2e-4
    top_truth_components: int = 256
    assignment_cost_limit: float = 9.0
    ad_weight: float = 1.0
    mass_weight: float = 0.65
    coordinate_weight: float = 0.18
    port_weight: float = 0.16
    residual_weight: float = 0.002
    seed: int = 20_260_817


def _label_path(paths: RuntimePaths) -> Path:
    return paths.work_dir / "training" / "path_state" / "official_labels.npz"


def _fold_state_path(paths: RuntimePaths, fold: int) -> Path:
    return paths.work_dir / "training" / "path_state" / f"fold_{fold}_state.npz"


def _oof_state_path(paths: RuntimePaths) -> Path:
    return paths.work_dir / "training" / "path_state" / "oof_state.npz"


def _fallback_targets(
    auxiliary: dict[str, np.ndarray], mask: np.ndarray, seed_count: int
) -> tuple[np.ndarray, np.ndarray]:
    fraction = np.asarray(auxiliary["fraction"], dtype=np.float32)
    presence = np.asarray(auxiliary["presence"], dtype=np.float32)
    mass = np.maximum(fraction, 1e-12) * np.sqrt(
        np.maximum(presence, 1.0) / float(seed_count)
    )
    mass *= mask
    mass /= np.maximum(mass.sum(axis=1, keepdims=True), TINY)
    port = np.asarray(auxiliary["port"], dtype=np.float32).reshape(
        len(mass), mass.shape[1], 8
    )
    port /= np.maximum(port.sum(axis=2, keepdims=True), TINY)
    return mass.astype(np.float32), port.astype(np.float32)


def _periodic_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return ((left - right + 0.5 * PERIODS) % PERIODS) - 0.5 * PERIODS


def build_path_labels(
    paths: RuntimePaths,
    pipeline_config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    config: PathTrainingConfig,
    force: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    output = _label_path(paths)
    if output.is_file() and not force:
        with np.load(output, allow_pickle=False) as loaded:
            labels = {name: np.array(loaded[name], copy=True) for name in loaded.files}
        return labels, {
            "status": "reused",
            "output": portable_path(output),
            "valid_count": int(np.sum(labels["valid"])),
        }

    started = time.time()
    data = load_official_data(paths.data_dir)
    features, auxiliary = load_model_inputs(paths, "train")
    del features
    mask_path = path_token_dir(paths, "train") / "mask.npy"
    if not mask_path.is_file():
        raise FileNotFoundError(
            f"missing Train path tokens: {mask_path}; run train-tokens first"
        )
    mask = np.asarray(np.load(mask_path, mmap_mode="r"), dtype=bool)
    coordinate = np.asarray(auxiliary["coordinate"], dtype=np.float32)
    fallback_mass, fallback_port = _fallback_targets(
        auxiliary, mask, pipeline_config.seed_count
    )
    count, candidates = mask.shape
    target_mass = np.zeros((count, candidates), dtype=np.float32)
    target_delta = np.zeros((count, candidates, 3), dtype=np.float32)
    target_port = fallback_port.copy()
    matched_fraction = np.zeros(count, dtype=np.float32)
    matched_count = np.zeros(count, dtype=np.int16)
    valid = np.zeros(count, dtype=bool)
    channel = data.train_channel(mmap_mode="r")
    scales = np.asarray((2.0, 1.5, 3.0), dtype=np.float32)

    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        block = torch.as_tensor(
            np.array(channel[start:end], dtype=np.complex64, copy=True), device=device
        ).reshape(end - start, 2, 16, 8, 4, 192)
        with torch.inference_mode():
            transformed = torch.fft.ifft(
                torch.fft.fft2(block, dim=(2, 3)), dim=-1
            )
            port_power = transformed.abs().square().float()
            collapsed = port_power.sum(dim=(1, 4))
            flat = collapsed.reshape(end - start, -1)
            top_count = min(config.top_truth_components, flat.shape[1])
            top_energy, top_index = torch.topk(flat, top_count, dim=1)
            total_power = flat.sum(dim=1)
            port_flat = port_power.permute(0, 2, 3, 5, 1, 4).reshape(
                end - start, -1, 8
            )
            selected_port = torch.gather(
                port_flat,
                1,
                top_index[..., None].expand(-1, -1, 8),
            )
        energy_np = top_energy.cpu().numpy().astype(np.float64)
        index_np = top_index.cpu().numpy().astype(np.int64)
        port_np = selected_port.cpu().numpy().astype(np.float64)
        total_np = total_power.cpu().numpy().astype(np.float64)
        del block, transformed, port_power, collapsed, flat, selected_port

        horizontal = index_np // (8 * 192)
        remainder = index_np % (8 * 192)
        vertical = remainder // 192
        delay = remainder % 192
        truth_coordinate = np.stack((horizontal, vertical, delay), axis=2).astype(
            np.float32
        )

        for local_row, global_row in enumerate(range(start, end)):
            candidate_ids = np.flatnonzero(mask[global_row])
            if not len(candidate_ids) or total_np[local_row] <= TINY:
                target_mass[global_row] = fallback_mass[global_row]
                continue
            candidate_coordinate = coordinate[global_row, candidate_ids]
            truth = truth_coordinate[local_row]
            delta = _periodic_delta(
                truth[:, None, :], candidate_coordinate[None, :, :]
            )
            cost = np.sum((delta / scales[None, None, :]) ** 2, axis=2)
            nearest = np.argmin(cost, axis=1)
            nearest_cost = cost[np.arange(len(truth)), nearest]
            accepted = nearest_cost <= config.assignment_cost_limit
            accepted &= energy_np[local_row] > 0.0
            if not np.any(accepted):
                target_mass[global_row] = fallback_mass[global_row]
                continue

            local_target = np.zeros(len(candidate_ids), dtype=np.float64)
            port_sum = np.zeros((len(candidate_ids), 8), dtype=np.float64)
            sine = np.zeros((len(candidate_ids), 3), dtype=np.float64)
            cosine = np.zeros((len(candidate_ids), 3), dtype=np.float64)
            assignments = nearest[accepted]
            accepted_energy = energy_np[local_row, accepted]
            accepted_truth = truth[accepted]
            accepted_port = port_np[local_row, accepted]
            np.add.at(local_target, assignments, accepted_energy)
            np.add.at(port_sum, assignments, accepted_port)
            phase = accepted_truth * (2.0 * np.pi / PERIODS)[None, :]
            np.add.at(sine, assignments, accepted_energy[:, None] * np.sin(phase))
            np.add.at(cosine, assignments, accepted_energy[:, None] * np.cos(phase))

            assigned = local_target > 0.0
            circular = np.mod(np.arctan2(sine, cosine), 2.0 * np.pi)
            center = circular * PERIODS[None, :] / (2.0 * np.pi)
            local_delta = _periodic_delta(center, candidate_coordinate)
            global_candidates = candidate_ids[assigned]
            target_delta[global_row, global_candidates] = local_delta[assigned]
            normalized_port = port_sum[assigned]
            normalized_port /= np.maximum(
                normalized_port.sum(axis=1, keepdims=True), TINY
            )
            target_port[global_row, global_candidates] = (
                0.9 * normalized_port
                + 0.1 * fallback_port[global_row, global_candidates]
            ).astype(np.float32)
            local_target /= max(float(np.sum(local_target)), TINY)
            blended = np.zeros(candidates, dtype=np.float32)
            blended[candidate_ids] = local_target.astype(np.float32)
            blended = 0.9 * blended + 0.1 * fallback_mass[global_row]
            blended *= mask[global_row]
            blended /= max(float(np.sum(blended)), TINY)
            target_mass[global_row] = blended
            matched_fraction[global_row] = np.float32(
                np.sum(accepted_energy) / max(total_np[local_row], TINY)
            )
            matched_count[global_row] = int(np.sum(assigned))
            valid[global_row] = True
        print(f"path labels {end}/{count}", flush=True)

    target_coordinate = np.mod(coordinate + target_delta, PERIODS).astype(np.float32)
    payload = {
        "mass": target_mass,
        "coordinate": target_coordinate,
        "delta": target_delta,
        "port": target_port.reshape(count, candidates, 2, 4),
        "matched_fraction": matched_fraction,
        "matched_count": matched_count,
        "valid": valid,
    }
    atomic_npz(output, **payload)
    report = {
        "status": "complete",
        "output": portable_path(output),
        "point_count": int(count),
        "valid_count": int(np.sum(valid)),
        "matched_fraction_mean": float(np.mean(matched_fraction)),
        "matched_fraction_nonzero_mean": float(
            np.mean(matched_fraction[matched_fraction > 0.0])
            if np.any(matched_fraction > 0.0)
            else 0.0
        ),
        "matched_count_mean": float(np.mean(matched_count)),
        "wall_seconds": float(time.time() - started),
    }
    atomic_json(paths.report_dir / "path_state_labels.json", report)
    return payload, report


def _feature_statistics(
    features: np.ndarray, mask: np.ndarray, point_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(features.shape[-1], dtype=np.float64)
    square = np.zeros(features.shape[-1], dtype=np.float64)
    count = 0
    for start in range(0, len(point_ids), 64):
        rows = point_ids[start : start + 64]
        local_mask = mask[rows]
        values = np.asarray(features[rows], dtype=np.float32)[local_mask]
        total += values.sum(axis=0, dtype=np.float64)
        square += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += len(values)
    if count == 0:
        raise RuntimeError("cannot standardize PathState features without valid paths")
    mean = total / count
    variance = np.maximum(square / count - mean * mean, 0.0)
    std = np.sqrt(variance)
    std[std < 1e-4] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _spectral_kernels(
    coordinate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    horizontal_index = torch.arange(16, device=coordinate.device, dtype=torch.float32)
    vertical_index = torch.arange(8, device=coordinate.device, dtype=torch.float32)
    delay_index = torch.arange(192, device=coordinate.device, dtype=torch.float32)
    horizontal = torch.fft.fft(
        torch.exp(
            2j
            * torch.pi
            * coordinate[..., 0, None]
            * horizontal_index
            / 16.0
        ),
        dim=-1,
    ).abs().square()
    vertical = torch.fft.fft(
        torch.exp(
            2j
            * torch.pi
            * coordinate[..., 1, None]
            * vertical_index
            / 8.0
        ),
        dim=-1,
    ).abs().square()
    delay = torch.fft.ifft(
        torch.exp(
            -2j
            * torch.pi
            * coordinate[..., 2, None]
            * delay_index
            / 192.0
        ),
        dim=-1,
    ).abs().square()
    horizontal = horizontal / horizontal.sum(dim=-1, keepdim=True).clamp_min(TINY)
    vertical = vertical / vertical.sum(dim=-1, keepdim=True).clamp_min(TINY)
    delay = delay / delay.sum(dim=-1, keepdim=True).clamp_min(TINY)
    return horizontal.float(), vertical.float(), delay.float()


def _objective(
    output: dict[str, torch.Tensor],
    target_mass: torch.Tensor,
    target_coordinate: torch.Tensor,
    target_port: torch.Tensor,
    truth_ad: torch.Tensor,
    config: PathTrainingConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    horizontal, vertical, delay = _spectral_kernels(output["coordinate"])
    rendered_ad = torch.einsum(
        "bk,bkh,bkv,bkd->bhvd", output["mass"], horizontal, vertical, delay
    )
    rendered_flat = rendered_ad.reshape(len(rendered_ad), -1)
    truth_flat = truth_ad.reshape(len(truth_ad), -1)
    ad_cosine = (rendered_flat * truth_flat).sum(dim=1) / torch.sqrt(
        rendered_flat.square().sum(dim=1)
        * truth_flat.square().sum(dim=1)
    ).clamp_min(TINY)
    mass_affinity = (
        torch.sqrt(output["mass"].clamp_min(1e-12))
        * torch.sqrt(target_mass.clamp_min(0.0))
    ).sum(dim=1)
    coordinate_delta = torch.remainder(
        output["coordinate"] - target_coordinate
        + 0.5 * torch.as_tensor(PERIODS, device=output["coordinate"].device),
        torch.as_tensor(PERIODS, device=output["coordinate"].device),
    ) - 0.5 * torch.as_tensor(PERIODS, device=output["coordinate"].device)
    coordinate_error = torch.nn.functional.smooth_l1_loss(
        coordinate_delta / torch.tensor((2.0, 1.5, 3.0), device=coordinate_delta.device),
        torch.zeros_like(coordinate_delta),
        reduction="none",
    ).mean(dim=2)
    coordinate_loss = (target_mass * coordinate_error).sum(dim=1)
    predicted_port = output["port"].reshape_as(target_port)
    port_error = -(
        target_port.clamp_min(0.0)
        * torch.log(predicted_port.clamp_min(1e-8))
    ).sum(dim=2)
    port_loss = (target_mass * port_error).sum(dim=1)
    residual = (
        output["score_residual"].square().mean()
        + output["coordinate_residual"].square().mean()
        + output["port_residual"].square().mean()
    )
    loss = (
        config.ad_weight * (1.0 - ad_cosine.mean())
        + config.mass_weight * (1.0 - mass_affinity.mean())
        + config.coordinate_weight * coordinate_loss.mean()
        + config.port_weight * port_loss.mean()
        + config.residual_weight * residual
    )
    return loss, {
        "ad_cosine": ad_cosine.mean(),
        "mass_affinity": mass_affinity.mean(),
        "coordinate_loss": coordinate_loss.mean(),
        "port_loss": port_loss.mean(),
    }


def _run_model(
    model: PathStateModel,
    rows: np.ndarray,
    features: np.ndarray,
    auxiliary: dict[str, np.ndarray],
    coordinate: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    seed_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return model(
        _batch(rows, features, auxiliary, coordinate, mean, std, seed_count, device)
    )


@torch.inference_mode()
def _evaluate_model(
    model: PathStateModel,
    point_ids: np.ndarray,
    features: np.ndarray,
    auxiliary: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    truth_ad: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    pipeline_config: PipelineConfig,
    config: PathTrainingConfig,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    totals: list[dict[str, float]] = []
    losses = []
    for start in range(0, len(point_ids), config.batch_size):
        rows = point_ids[start : start + config.batch_size]
        output = _run_model(
            model,
            rows,
            features,
            auxiliary,
            auxiliary["coordinate"],
            mean,
            std,
            pipeline_config.seed_count,
            device,
        )
        loss, metrics = _objective(
            output,
            torch.as_tensor(labels["mass"][rows], device=device),
            torch.as_tensor(labels["coordinate"][rows], device=device),
            torch.as_tensor(
                labels["port"][rows].reshape(len(rows), -1, 8), device=device
            ),
            torch.as_tensor(
                np.asarray(truth_ad[rows], dtype=np.float32), device=device
            ),
            config,
        )
        losses.append(float(loss.cpu()))
        totals.append({name: float(value.cpu()) for name, value in metrics.items()})
    summary = {
        name: float(np.mean([row[name] for row in totals])) for name in totals[0]
    }
    return float(np.mean(losses)), summary


def _train_epochs(
    train_ids: np.ndarray,
    validation_ids: np.ndarray | None,
    epochs: int,
    features: np.ndarray,
    auxiliary: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    truth_ad: np.ndarray,
    mask: np.ndarray,
    pipeline_config: PipelineConfig,
    config: PathTrainingConfig,
    model_config: ModelConfig,
    device: torch.device,
    seed: int,
) -> tuple[PathStateModel, np.ndarray, np.ndarray, int, list[dict[str, Any]]]:
    set_seed(seed)
    mean, std = _feature_statistics(features, mask, train_ids)
    model = PathStateModel(
        features.shape[-1], auxiliary["geometry"].shape[-1], model_config
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=config.learning_rate * 0.1
    )
    rng = np.random.default_rng(seed)
    best_epoch = max(1, min(config.minimum_epochs, epochs))
    best_loss = float("inf")
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    history: list[dict[str, Any]] = []
    train_ids = train_ids[np.asarray(labels["valid"])[train_ids]]
    if not len(train_ids):
        raise RuntimeError("PathState training fold has no valid labels")

    for epoch in range(1, epochs + 1):
        model.train()
        batch_rows = rng.permutation(train_ids)
        epoch_metrics: list[dict[str, float]] = []
        for start in range(0, len(batch_rows), config.batch_size):
            rows = batch_rows[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = _run_model(
                model,
                rows,
                features,
                auxiliary,
                auxiliary["coordinate"],
                mean,
                std,
                pipeline_config.seed_count,
                device,
            )
            loss, metrics = _objective(
                output,
                torch.as_tensor(labels["mass"][rows], device=device),
                torch.as_tensor(labels["coordinate"][rows], device=device),
                torch.as_tensor(
                    labels["port"][rows].reshape(len(rows), -1, 8),
                    device=device,
                ),
                torch.as_tensor(
                    np.asarray(truth_ad[rows], dtype=np.float32), device=device
                ),
                config,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_metrics.append(
                {"loss": float(loss.detach().cpu())}
                | {name: float(value.detach().cpu()) for name, value in metrics.items()}
            )
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            **{
                name: float(np.mean([value[name] for value in epoch_metrics]))
                for name in epoch_metrics[0]
            },
        }
        if (
            validation_ids is not None
            and epoch >= config.minimum_epochs
            and (
                epoch % config.validation_interval == 0
                or epoch == epochs
            )
        ):
            validation = validation_ids[
                np.asarray(labels["valid"])[validation_ids]
            ]
            validation_loss, validation_metrics = _evaluate_model(
                model,
                validation,
                features,
                auxiliary,
                labels,
                truth_ad,
                mean,
                std,
                pipeline_config,
                config,
                device,
            )
            row["validation_loss"] = validation_loss
            row["validation"] = validation_metrics
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
        history.append(row)
        print(
            f"PathState epoch={epoch}/{epochs} loss={row['loss']:.6f} "
            f"val={row.get('validation_loss', float('nan')):.6f}",
            flush=True,
        )
    if validation_ids is None:
        best_epoch = epochs
        best_state = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
    model.load_state_dict(best_state)
    return model, mean, std, best_epoch, history


@torch.inference_mode()
def _predict_state(
    model: PathStateModel,
    point_ids: np.ndarray,
    features: np.ndarray,
    auxiliary: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    pipeline_config: PipelineConfig,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    blocks: list[dict[str, np.ndarray]] = []
    for start in range(0, len(point_ids), batch_size):
        rows = point_ids[start : start + batch_size]
        output = _run_model(
            model,
            rows,
            features,
            auxiliary,
            auxiliary["coordinate"],
            mean,
            std,
            pipeline_config.seed_count,
            device,
        )
        blocks.append(
            {
                name: output[name].float().cpu().numpy()
                for name in ("mass", "coordinate", "port")
            }
        )
    return {
        name: np.concatenate([block[name] for block in blocks], axis=0)
        for name in ("mass", "coordinate", "port")
    }


def train_path_state_models(
    paths: RuntimePaths,
    pipeline_config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool = False,
    training_config: PathTrainingConfig | None = None,
) -> dict[str, Any]:
    started = time.time()
    config = training_config or PathTrainingConfig(batch_size=batch_size)
    _, layout, preparation = prepare_training_inputs(
        paths, pipeline_config, device, batch_size, force=False
    )
    folds = load_fold_metadata(layout.fold_cache, layout.fold_metadata)
    features, auxiliary = load_model_inputs(paths, "train")
    mask = np.asarray(
        np.load(path_token_dir(paths, "train") / "mask.npy", mmap_mode="r"),
        dtype=bool,
    )
    labels, label_report = build_path_labels(
        paths, pipeline_config, device, min(batch_size, 16), config, force=force
    )
    truth_ad = np.load(paths.base_dir / "official_spectra" / "angle_delay.npy", mmap_mode="r")
    model_config = ModelConfig()
    lookup = folds.query_lookup
    fold_reports = []

    for fold in range(5):
        checkpoint = checkpoint_path(paths.model_dir, fold)
        state_path = _fold_state_path(paths, fold)
        if checkpoint.is_file() and state_path.is_file() and not force:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            fold_reports.append(
                {
                    "fold": fold,
                    "status": "reused",
                    "checkpoint": portable_path(checkpoint),
                    "best_epoch": int(saved.get("best_epoch", -1)),
                }
            )
            continue

        outer_fit = folds.fit_indices[fold]
        outer_query = folds.query_indices[fold]
        inner = (fold + 1) % 5
        nested_fit = np.intersect1d(
            outer_fit, folds.fit_indices[inner], assume_unique=False
        )
        nested_validation = np.intersect1d(
            outer_fit, folds.query_indices[inner], assume_unique=False
        )
        screening, _, _, best_epoch, screen_history = _train_epochs(
            nested_fit,
            nested_validation,
            config.epochs,
            features,
            auxiliary,
            labels,
            truth_ad,
            mask,
            pipeline_config,
            config,
            model_config,
            device,
            config.seed + fold * 1000,
        )
        del screening
        if device.type == "cuda":
            torch.cuda.empty_cache()
        final_model, mean, std, _, refit_history = _train_epochs(
            outer_fit,
            None,
            best_epoch,
            features,
            auxiliary,
            labels,
            truth_ad,
            mask,
            pipeline_config,
            config,
            model_config,
            device,
            config.seed + 10_000 + fold * 1000,
        )
        held_loss, held_metrics = _evaluate_model(
            final_model,
            outer_query[np.asarray(labels["valid"])[outer_query]],
            features,
            auxiliary,
            labels,
            truth_ad,
            mean,
            std,
            pipeline_config,
            config,
            device,
        )
        state = _predict_state(
            final_model,
            outer_query,
            features,
            auxiliary,
            mean,
            std,
            pipeline_config,
            config.batch_size,
            device,
        )
        atomic_npz(
            state_path,
            query_ids=outer_query.astype(np.int32),
            mass=state["mass"].astype(np.float32),
            coordinate=np.mod(state["coordinate"], PERIODS).astype(np.float32),
            port=state["port"].astype(np.float32),
        )
        checkpoint_config = asdict(model_config) | asdict(config) | {
            "outer": fold,
            "validation_fold": inner,
        }
        atomic_torch_save(
            checkpoint,
            {
                "state_dict": final_model.state_dict(),
                "config": checkpoint_config,
                "feature_mean": mean,
                "feature_std": std,
                "best_epoch": int(best_epoch),
                "outer": int(fold),
                "mapping_latent": np.zeros(10, dtype=np.float32),
            },
        )
        fold_reports.append(
            {
                "fold": fold,
                "status": "trained",
                "fit_count": int(len(outer_fit)),
                "held_count": int(len(outer_query)),
                "best_epoch": int(best_epoch),
                "held_loss": float(held_loss),
                "held": held_metrics,
                "checkpoint": portable_path(checkpoint),
                "state": portable_path(state_path),
                "screen_history": screen_history,
                "refit_history": refit_history,
            }
        )
        del final_model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof_mass = np.empty((len(folds.query_ids), mask.shape[1]), dtype=np.float32)
    oof_coordinate = np.empty((len(folds.query_ids), mask.shape[1], 3), dtype=np.float32)
    oof_port = np.empty((len(folds.query_ids), mask.shape[1], 2, 4), dtype=np.float32)
    for fold in range(5):
        with np.load(_fold_state_path(paths, fold), allow_pickle=False) as state:
            rows = lookup[state["query_ids"].astype(np.int64)]
            oof_mass[rows] = state["mass"].astype(np.float32)
            oof_coordinate[rows] = state["coordinate"].astype(np.float32)
            oof_port[rows] = state["port"].astype(np.float32)
    atomic_npz(
        _oof_state_path(paths),
        query_ids=folds.query_ids.astype(np.int32),
        fold_id=folds.fold_id,
        mass=oof_mass,
        coordinate=oof_coordinate,
        port=oof_port,
    )
    report = {
        "stage": "train_path_state",
        "status": "complete",
        "config": asdict(config),
        "model_config": asdict(model_config),
        "preparation": preparation,
        "labels": label_report,
        "folds": fold_reports,
        "oof_state": portable_path(_oof_state_path(paths)),
        "wall_seconds": float(time.time() - started),
    }
    atomic_json(paths.report_dir / "train_path_state.json", report)
    return report
