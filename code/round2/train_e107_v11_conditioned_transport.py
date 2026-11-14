#!/usr/bin/env python3
"""Fine-tune E104 transport heads through the frozen V12 channel projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np
import torch

import analyze_e100_locked_energy_angle_correction as e100
import analyze_e86_surface_endpoint_v11_blend as e86
import train_e104_locked_pdp_angle_transport as e104
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics
from train_full_path_pas_slice_gate import project_joint


VERSION = "Round2-E107-v11-conditioned-transport"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v11_conditioned_transport"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v11_conditioned_transport"
REPORT = ANALYSIS_DIR / "v11_conditioned_transport.json"
METRICS = ARTIFACT_DIR / "strict_oof_v11_conditioned_transport_metrics.npz"
OOF_PAS = ARTIFACT_DIR / "strict_oof_v11_conditioned_transport_pas_f16.npy"
OLD_PAS = e85.ARTIFACT_DIR / "e86_geometry_sequence_rendered_pas_f16.npy"
OLD_PDP = e85.ARTIFACT_DIR / "e86_geometry_sequence_rendered_pdp_f16.npy"
TINY = 1e-30


class StableSqrt(torch.autograd.Function):
    """Keep the exact sqrt forward value while bounding its derivative at zero."""

    @staticmethod
    def forward(ctx: object, value: torch.Tensor, epsilon: float) -> torch.Tensor:
        ctx.save_for_backward(value)
        ctx.epsilon = epsilon
        return torch.sqrt(value.clamp_min(0.0))

    @staticmethod
    def backward(
        ctx: object, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (value,) = ctx.saved_tensors
        denominator = 2.0 * torch.sqrt(value.clamp_min(0.0) + ctx.epsilon)
        return gradient / denominator, None


def stable_sqrt(value: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    return StableSqrt.apply(value, epsilon)


def stable_endpoint_target(
    base: torch.Tensor,
    endpoint: torch.Tensor,
    beta: float,
    dims: tuple[int, ...],
) -> torch.Tensor:
    mixed = (
        (1.0 - beta) * stable_sqrt(base)
        + beta * stable_sqrt(endpoint)
    ).square()
    total = mixed.sum(dim=dims, keepdim=True)
    return torch.where(
        total > TINY, mixed / total.clamp_min(TINY), torch.zeros_like(mixed)
    )


def stable_pas_projection(
    channel: torch.Tensor, desired: torch.Tensor
) -> torch.Tensor:
    shaped = channel.reshape(len(channel), 2, 16, 8, 4, 192)
    spectrum = torch.fft.fft2(shaped, dim=(2, 3))
    magnitude = spectrum.abs()
    slice_energy = magnitude.square().sum(dim=(1, 2, 3), keepdim=True)
    desired_magnitude = stable_sqrt((desired * slice_energy).clamp_min(0.0))
    unit = spectrum / magnitude.clamp_min(TINY)
    unit = torch.where(
        magnitude > TINY,
        unit,
        torch.ones((), dtype=spectrum.dtype, device=spectrum.device),
    )
    return torch.fft.ifft2(desired_magnitude * unit, dim=(2, 3)).reshape_as(channel)


def stable_pdp_projection(
    channel: torch.Tensor, desired: torch.Tensor
) -> torch.Tensor:
    delay = torch.fft.ifft(channel, dim=-1)
    magnitude = delay.abs()
    link_energy = magnitude.square().sum(dim=3, keepdim=True)
    desired_magnitude = stable_sqrt((desired * link_energy).clamp_min(0.0))
    unit = delay / magnitude.clamp_min(TINY)
    unit = torch.where(
        magnitude > TINY,
        unit,
        torch.ones((), dtype=delay.dtype, device=delay.device),
    )
    return torch.fft.fft(desired_magnitude * unit, dim=-1)


def stable_preserve_power(
    channel: torch.Tensor, reference_power: torch.Tensor
) -> torch.Tensor:
    current = channel.abs().square().sum(dim=(1, 2, 3))
    scale = stable_sqrt(reference_power / current.clamp_min(TINY))
    return channel * scale[:, None, None, None]


def stable_project_joint(
    base: torch.Tensor,
    desired_pas: torch.Tensor,
    desired_pdp: torch.Tensor,
    cycles: int,
    order: str,
) -> torch.Tensor:
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    output = base
    for _ in range(cycles):
        for domain in order.split("_"):
            if domain == "pas":
                output = stable_pas_projection(output, desired_pas)
            elif domain == "pdp":
                output = stable_pdp_projection(output, desired_pdp)
            else:
                raise ValueError(domain)
        output = stable_preserve_power(output, reference_power)
    return output


@dataclass(frozen=True)
class AdaptConfig:
    pretrain_epochs: int = 4
    adapt_epochs: int = 12
    validation_interval: int = 1
    batch_size: int = 12
    evaluation_batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 2e-4
    gate_penalty: float = 0.001
    random_state: int = 20260808107
    pas_beta: float = 0.20
    pdp_beta: float = 0.20
    cycles: int = 4
    order: str = "pas_pdp"


class CanonicalStore:
    def __init__(self, query_ids: np.ndarray) -> None:
        self.query_ids = query_ids
        self.lookup = np.full(4000, -1, dtype=np.int32)
        self.lookup[query_ids] = np.arange(len(query_ids), dtype=np.int32)
        self.base = np.load(e86.V11_CHANNEL, mmap_mode="r")
        self.truth = np.load(e86.TRAIN_CHANNEL, mmap_mode="r")
        self.endpoint_pdp = np.load(OLD_PDP, mmap_mode="r")
        self.v12 = np.load(e100.V12_CHANNEL, mmap_mode="r")

    def local(self, global_rows: np.ndarray) -> np.ndarray:
        rows = self.lookup[global_rows]
        if np.any(rows < 0):
            raise ValueError("rows are not part of the canonical 500 points")
        return rows.astype(np.int64)

    def tensors(
        self, global_rows: np.ndarray, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local = self.local(global_rows)
        base = torch.as_tensor(
            np.asarray(self.base[local], dtype=np.complex64), device=device
        )
        truth = torch.as_tensor(
            np.asarray(self.truth[global_rows], dtype=np.complex64), device=device
        )
        endpoint_pdp = torch.as_tensor(
            np.asarray(self.endpoint_pdp[local], dtype=np.float32), device=device
        )
        return base, truth, endpoint_pdp


def cosine(
    prediction: torch.Tensor, truth: torch.Tensor, dims: tuple[int, ...]
) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=dims, dtype=torch.float64)
    denominator_squared = (
        prediction.square().sum(dim=dims, dtype=torch.float64)
        * truth.square().sum(dim=dims, dtype=torch.float64)
    )
    denominator = stable_sqrt(denominator_squared, 1e-24)
    return numerator / torch.where(
        denominator_squared == 0.0, 1e-12, denominator
    )


def differentiable_score(
    prediction: torch.Tensor, truth: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction_pas = pas_distribution(prediction)
    truth_pas = pas_distribution(truth)
    prediction_pdp = pdp_distribution(prediction)
    truth_pdp = pdp_distribution(truth)
    pas = cosine(prediction_pas, truth_pas, (1, 2, 3)).mean(
        dim=(1, 2), dtype=torch.float64
    )
    pdp = cosine(prediction_pdp, truth_pdp, (3,)).mean(
        dim=(1, 2), dtype=torch.float64
    )
    error = (prediction - truth).abs().square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    energy = truth.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    nmse = error.sum() / energy.sum().clamp_min(TINY)
    nmse_score = 1.0 / (1.0 + nmse)
    score = 0.4 * pas.mean() + 0.4 * pdp.mean() + 0.2 * nmse_score
    return score, {
        "pas": pas.mean(),
        "pdp": pdp.mean(),
        "nmse": nmse,
        "nmse_score": nmse_score,
    }


def candidate_channel(
    model: e104.LockedPdpTransport,
    endpoint_store: e85.DataStore,
    endpoint: e104.EndpointData,
    canonical: CanonicalStore,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    valid: np.ndarray,
    config: AdaptConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = e104.endpoint_features(
        endpoint_store, endpoint, rows, mean, std, valid, device
    )
    endpoint_pas, endpoint_score, alpha = e104.predict_batch(model, batch)
    base, truth, endpoint_pdp = canonical.tensors(rows, device)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    desired_pas = stable_endpoint_target(
        base_pas,
        endpoint_pas[..., None].expand(-1, -1, -1, -1, -1, 192),
        config.pas_beta,
        (1, 2, 3),
    )
    desired_pdp = stable_endpoint_target(
        base_pdp, endpoint_pdp, config.pdp_beta, (3,)
    )
    channel = stable_project_joint(
        base, desired_pas, desired_pdp, config.cycles, config.order
    )
    return channel, truth, torch.stack((endpoint_score.mean(), alpha.mean()))


def trainable_transport_heads(
    model: e104.LockedPdpTransport,
) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = []
    for module in (model.query, model.key, model.gate):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            parameters.append(parameter)
    return parameters


def train_adapt_epoch(
    model: e104.LockedPdpTransport,
    optimizer: torch.optim.Optimizer,
    endpoint_store: e85.DataStore,
    endpoint: e104.EndpointData,
    canonical: CanonicalStore,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    valid: np.ndarray,
    config: AdaptConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.eval()
    order = rng.permutation(rows)
    score_total = 0.0
    endpoint_total = 0.0
    gate_total = 0.0
    count = 0
    for start in range(0, len(order), config.batch_size):
        local_rows = order[start : start + config.batch_size]
        channel, truth, auxiliary = candidate_channel(
            model,
            endpoint_store,
            endpoint,
            canonical,
            local_rows,
            mean,
            std,
            valid,
            config,
            device,
        )
        score, _ = differentiable_score(channel, truth)
        loss = 1.0 - score + config.gate_penalty * auxiliary[1].square()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError("non-finite E107 gradient")
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()
        count += len(local_rows)
        score_total += float(score.detach().item()) * len(local_rows)
        endpoint_total += float(auxiliary[0].detach().item()) * len(local_rows)
        gate_total += float(auxiliary[1].detach().item()) * len(local_rows)
    return {
        "score": score_total / max(count, 1),
        "endpoint_pas": endpoint_total / max(count, 1),
        "gate": gate_total / max(count, 1),
    }


@torch.inference_mode()
def evaluate_model(
    model: e104.LockedPdpTransport,
    endpoint_store: e85.DataStore,
    endpoint: e104.EndpointData,
    canonical: CanonicalStore,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    valid: np.ndarray,
    config: AdaptConfig,
    device: torch.device,
    collect_pas: bool,
) -> dict[str, object]:
    model.eval()
    metrics: list[list[np.ndarray]] = [[], [], [], []]
    endpoint_pas = []
    gates = []
    for start in range(0, len(rows), config.evaluation_batch_size):
        local_rows = rows[start : start + config.evaluation_batch_size]
        batch = e104.endpoint_features(
            endpoint_store, endpoint, local_rows, mean, std, valid, device
        )
        pas, _, alpha = e104.predict_batch(model, batch)
        base, truth, endpoint_pdp = canonical.tensors(local_rows, device)
        desired_pas = e86.endpoint_target(
            pas_distribution(base),
            pas[..., None].expand(-1, -1, -1, -1, -1, 192),
            config.pas_beta,
            "sqrt",
            (1, 2, 3),
        )
        desired_pdp = e86.endpoint_target(
            pdp_distribution(base),
            endpoint_pdp,
            config.pdp_beta,
            "sqrt",
            (3,),
        )
        channel = project_joint(
            base, desired_pas, desired_pdp, config.cycles, config.order
        )
        values = point_metrics(
            channel,
            truth,
            pas_distribution(truth),
            pdp_distribution(truth),
        )
        for destination, source in zip(metrics, values):
            destination.append(source)
        gates.append(alpha.mean(dim=1).cpu().numpy())
        if collect_pas:
            endpoint_pas.append(pas.cpu().numpy().astype(np.float32))
    values = tuple(np.concatenate(parts) for parts in metrics)
    return {
        "metrics": values,
        "summary": statistics(*values),
        "gate": np.concatenate(gates),
        "endpoint_pas": np.concatenate(endpoint_pas) if collect_pas else None,
    }


def pretrain(
    endpoint_store: e85.DataStore,
    endpoint: e104.EndpointData,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    valid: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
) -> e104.LockedPdpTransport:
    model, _, _ = e104.train_model(
        endpoint_store,
        endpoint,
        rows,
        None,
        None,
        mean,
        std,
        valid,
        e104.Config(),
        epochs,
        seed,
        device,
    )
    return model


def fit_adaptation(
    model: e104.LockedPdpTransport,
    endpoint_store: e85.DataStore,
    endpoint: e104.EndpointData,
    canonical: CanonicalStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray | None,
    mean: np.ndarray,
    std: np.ndarray,
    valid: np.ndarray,
    config: AdaptConfig,
    epochs: int,
    seed: int,
    device: torch.device,
    validation_baseline: dict[str, float | int] | None,
) -> tuple[list[dict[str, object]], int]:
    trainable = trainable_transport_heads(model)
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(seed)
    history = []
    best_epoch = -1
    best_score = (
        -math.inf
        if validation_baseline is None
        else float(validation_baseline["score"])
    )
    if validation_rows is not None:
        initial = evaluate_model(
            model,
            endpoint_store,
            endpoint,
            canonical,
            validation_rows,
            mean,
            std,
            valid,
            config,
            device,
            False,
        )
        history.append(
            {
                "epoch": 0,
                "validation": initial["summary"],
                "identity_baseline": validation_baseline,
            }
        )
        if float(initial["summary"]["score"]) > best_score:
            best_score = float(initial["summary"]["score"])
            best_epoch = 0
    for epoch in range(1, epochs + 1):
        train = train_adapt_epoch(
            model,
            optimizer,
            endpoint_store,
            endpoint,
            canonical,
            fit_rows,
            mean,
            std,
            valid,
            config,
            device,
            rng,
        )
        row: dict[str, object] = {"epoch": epoch, "train": train}
        if validation_rows is not None and epoch % config.validation_interval == 0:
            validation = evaluate_model(
                model,
                endpoint_store,
                endpoint,
                canonical,
                validation_rows,
                mean,
                std,
                valid,
                config,
                device,
                False,
            )
            row["validation"] = validation["summary"]
            if float(validation["summary"]["score"]) > best_score:
                best_score = float(validation["summary"]["score"])
                best_epoch = epoch
        history.append(row)
        print(
            f"E107 adapt epoch={epoch}/{epochs} train={train['score']:.6f} "
            f"validation={row.get('validation', {}).get('score', float('nan')):.6f}",
            flush=True,
        )
    return history, best_epoch


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = AdaptConfig()
    required = (
        e104.FULL_ENDPOINT,
        e104.STRICT_ENDPOINT,
        e104.VALID_MASK,
        e86.V11_CHANNEL,
        e86.TRAIN_CHANNEL,
        e100.V12_CHANNEL,
        OLD_PAS,
        OLD_PDP,
    )
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    endpoint_store = e85.DataStore()
    full_endpoint = e104.EndpointData(e104.FULL_ENDPOINT, strict=False)
    strict_endpoint = e104.EndpointData(e104.STRICT_ENDPOINT, strict=True)
    query_ids = endpoint_store.strict_ids
    fold_id = endpoint_store.strict_fold
    if not np.array_equal(strict_endpoint.query_ids, query_ids):
        raise ValueError("strict endpoint order does not match canonical 500 points")
    canonical = CanonicalStore(query_ids)
    valid = np.asarray(np.load(e104.VALID_MASK, mmap_mode="r"), dtype=bool)
    all_rows = np.arange(len(endpoint_store.base), dtype=np.int64)

    truth = torch.as_tensor(
        np.asarray(canonical.truth[query_ids], dtype=np.complex64), device=device
    )
    v12 = torch.as_tensor(
        np.array(canonical.v12, dtype=np.complex64, copy=True), device=device
    )
    with torch.inference_mode():
        v12_metrics = point_metrics(
            v12, truth, pas_distribution(truth), pdp_distribution(truth)
        )
    del truth, v12

    selected_metrics = [np.zeros(len(query_ids), dtype=np.float64) for _ in range(4)]
    selected_pas = np.zeros(
        (len(query_ids), e85.MP, e85.MH, e85.MV, e85.N_UE), dtype=np.float32
    )
    selected_gate = np.zeros(len(query_ids), dtype=np.float32)
    old_pas = np.load(OLD_PAS, mmap_mode="r")
    training = []

    for outer in range(5):
        validation_fold = (outer + 1) % 5
        held = query_ids[fold_id == outer]
        validation = query_ids[fold_id == validation_fold]
        screen_canonical = query_ids[
            (fold_id != outer) & (fold_id != validation_fold)
        ]
        refit_canonical = query_ids[fold_id != outer]
        held_mask = np.zeros(len(endpoint_store.base), dtype=bool)
        validation_mask = np.zeros(len(endpoint_store.base), dtype=bool)
        held_mask[held] = True
        validation_mask[validation] = True
        screen_rows = all_rows[~held_mask & ~validation_mask]
        refit_rows = all_rows[~held_mask]

        mean_screen, std_screen = endpoint_store.normalization(screen_rows)
        screen_model = pretrain(
            endpoint_store,
            full_endpoint,
            screen_rows,
            mean_screen,
            std_screen,
            valid,
            config.pretrain_epochs,
            config.random_state + 10000 * outer,
            device,
        )
        validation_local = canonical.local(validation)
        validation_baseline = statistics(
            *(value[validation_local] for value in v12_metrics)
        )
        screen_history, best_epoch = fit_adaptation(
            screen_model,
            endpoint_store,
            strict_endpoint,
            canonical,
            screen_canonical,
            validation,
            mean_screen,
            std_screen,
            valid,
            config,
            config.adapt_epochs,
            config.random_state + 20000 * outer,
            device,
            validation_baseline,
        )
        del screen_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        local_held = np.flatnonzero(fold_id == outer)
        if best_epoch < 0:
            for destination, source in zip(selected_metrics, v12_metrics):
                destination[local_held] = source[local_held]
            selected_pas[local_held] = np.asarray(
                old_pas[local_held, ..., 0], dtype=np.float32
            )
            selected_gate[local_held] = 0.0
            held_summary = statistics(
                *(value[local_held] for value in v12_metrics)
            )
            checkpoint = None
            refit_history: list[dict[str, object]] = []
        else:
            mean_refit, std_refit = endpoint_store.normalization(refit_rows)
            refit_model = pretrain(
                endpoint_store,
                full_endpoint,
                refit_rows,
                mean_refit,
                std_refit,
                valid,
                config.pretrain_epochs,
                config.random_state + 100000 + 10000 * outer,
                device,
            )
            refit_history, _ = fit_adaptation(
                refit_model,
                endpoint_store,
                strict_endpoint,
                canonical,
                refit_canonical,
                None,
                mean_refit,
                std_refit,
                valid,
                config,
                best_epoch,
                config.random_state + 120000 + 10000 * outer,
                device,
                None,
            )
            prediction = evaluate_model(
                refit_model,
                endpoint_store,
                strict_endpoint,
                canonical,
                held,
                mean_refit,
                std_refit,
                valid,
                config,
                device,
                True,
            )
            for destination, source in zip(selected_metrics, prediction["metrics"]):
                destination[local_held] = source
            selected_pas[local_held] = prediction["endpoint_pas"]
            selected_gate[local_held] = prediction["gate"]
            held_summary = prediction["summary"]
            checkpoint_path = ARTIFACT_DIR / f"strict_fold{outer}_conditioned.pt"
            torch.save(
                {
                    "state_dict": refit_model.state_dict(),
                    "transport_config": asdict(e104.Config()),
                    "adapt_config": asdict(config),
                    "feature_mean": mean_refit,
                    "feature_std": std_refit,
                    "pretrain_epochs": config.pretrain_epochs,
                    "adapt_epochs": best_epoch,
                    "outer": outer,
                },
                checkpoint_path,
            )
            checkpoint = PATHS.relative(checkpoint_path)
            del refit_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        baseline_summary = statistics(
            *(value[local_held] for value in v12_metrics)
        )
        training.append(
            {
                "outer": outer,
                "validation_fold": validation_fold,
                "screen_pretrain_count": int(len(screen_rows)),
                "screen_adapt_count": int(len(screen_canonical)),
                "refit_pretrain_count": int(len(refit_rows)),
                "refit_adapt_count": int(len(refit_canonical)),
                "best_adapt_epoch": best_epoch,
                "screen_history": screen_history,
                "refit_history": refit_history,
                "held": held_summary,
                "held_baseline": baseline_summary,
                "held_delta": float(held_summary["score"] - baseline_summary["score"]),
                "checkpoint": checkpoint,
            }
        )
        print(
            f"E107 outer={outer} best_epoch={best_epoch} "
            f"held_delta={training[-1]['held_delta']:.6f}",
            flush=True,
        )

    selected_tuple = tuple(selected_metrics)
    comparison = e100.metric_comparison(selected_tuple, v12_metrics, fold_id)
    decision = {
        "minimum_score_gain": 0.001,
        "minimum_positive_folds": 4,
        "minimum_worst_fold_delta": -0.001,
        "score_gain": comparison["delta"],
        "positive_folds": comparison["positive_folds"],
        "worst_fold_delta": comparison["worst_fold_delta"],
    }
    decision["passed"] = bool(
        decision["score_gain"] >= decision["minimum_score_gain"]
        and decision["positive_folds"] >= decision["minimum_positive_folds"]
        and decision["worst_fold_delta"] >= decision["minimum_worst_fold_delta"]
    )
    np.save(OOF_PAS, selected_pas.astype(np.float16))
    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        point_pas=selected_metrics[0].astype(np.float32),
        point_pdp=selected_metrics[1].astype(np.float32),
        point_error=selected_metrics[2].astype(np.float64),
        point_energy=selected_metrics[3].astype(np.float64),
        endpoint_gate=selected_gate,
    )
    trainable_count = 2 * e104.Config().hidden * e104.Config().rank + e104.Config().hidden + 1
    report = {
        "experiment": VERSION,
        "status": "strict_complete",
        "primary_hypothesis": (
            "The E104 physical representation becomes deployable when only its small "
            "transport heads are fine-tuned through the actual V11/V12 complete score."
        ),
        "validation": (
            "Canonical 500 contiguous holes. Screen pretraining excludes both held and "
            "validation holes; the alternate hole selects adaptation epoch."
        ),
        "config": {
            "transport": asdict(e104.Config()),
            "adapt": asdict(config),
            "trainable_head_parameters": int(trainable_count),
            "frozen_encoder": True,
        },
        "training": training,
        "complete_score": {"versus_v12": comparison},
        "decision": {**decision, "generate_v13": decision["passed"]},
        "outputs": {
            "oof_pas": PATHS.relative(OOF_PAS),
            "metrics": PATHS.relative(METRICS),
            "pas_shape": list(selected_pas.shape),
            "pas_dtype": "float16",
            "finite": bool(
                np.isfinite(selected_pas).all()
                and np.isfinite(selected_gate).all()
                and all(np.isfinite(value).all() for value in selected_metrics)
            ),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "complete_score": report["complete_score"],
                "decision": report["decision"],
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
