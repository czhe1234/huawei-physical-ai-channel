#!/usr/bin/env python3
"""Train a zero-initialized path-mass residual directly around the V7 carrier."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import analyze_e119_v12_unique_path_complement as e119
import analyze_e131_v7_path_selection_evidence as e131
import analyze_e148_v7_pathwise_neural_residual as e148
import train_e115_complete_path_state_operator as e115
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import statistics


VERSION = "Round2-E149-v7-inloop-mass-residual"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v7_inloop_mass_residual"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v7_inloop_mass_residual"
METRICS = ARTIFACT_DIR / "v7_inloop_mass_residual_metrics.npz"
REPORT = ANALYSIS_DIR / "v7_inloop_mass_residual.json"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    hidden: int = 128
    heads: int = 8
    layers: int = 2
    dropout: float = 0.05
    residual_limit: float = 0.50
    epochs: int = 24
    validation_interval: int = 2
    batch_size: int = 12
    feature_batch_size: int = 24
    metric_batch_size: int = 12
    learning_rate: float = 4e-4
    weight_decay: float = 2e-4
    trust_weight: float = 1e-3
    component_barrier_weight: float = 0.50
    gradient_clip: float = 1.0
    required_score_gain: float = 0.005
    required_positive_folds: int = 4
    minimum_geometry_margin: float = 0.002
    maximum_worst_fold_loss: float = 0.001
    control_distance_strata: int = 4
    seed: int = 20_260_809_149


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--feature-batch-size", type=int, default=24)
    parser.add_argument("--metric-batch-size", type=int, default=12)
    return parser.parse_args()


class V7InLoopMassResidual(nn.Module):
    def __init__(self, evidence_dim: int, config: Config) -> None:
        super().__init__()
        hidden = config.hidden
        self.config = config
        self.evidence = nn.Sequential(
            nn.Linear(evidence_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.input_norm = nn.LayerNorm(hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=3 * hidden,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.output_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        frozen_hidden: torch.Tensor,
        evidence: torch.Tensor,
        mask: torch.Tensor,
        base_mass: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_norm(frozen_hidden + self.evidence(evidence))
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        hidden = self.output_norm(hidden)
        residual = self.config.residual_limit * torch.tanh(self.head(hidden)[..., 0])
        residual = residual.masked_fill(~mask, 0.0)
        center = (residual * base_mass).sum(dim=1, keepdim=True)
        return (residual - center).masked_fill(~mask, 0.0)


def raw_batch(
    store,
    point_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "coordinate": store.coordinate[point_ids],
        "base_log_mass": store.base_log_mass[point_ids],
        "base_log_port": store.base_log_port[point_ids],
        "base_phase": store.base_phase[point_ids],
        "base_log_power": store.raw.base_log_power[point_ids],
        "mask": store.mask[point_ids],
    }


@torch.inference_mode()
def extract_mass_hidden(
    model,
    store,
    normalization: dict[str, torch.Tensor],
    point_ids: np.ndarray,
    batch_size: int,
) -> torch.Tensor:
    output = torch.empty(
        (len(point_ids), store.mask.shape[1], model.config.hidden),
        dtype=torch.float32,
        device=store.device,
    )
    captured: list[torch.Tensor] = []

    def capture(_module, inputs) -> None:
        captured.append(inputs[0])

    handle = model.score_head.register_forward_pre_hook(capture)
    try:
        for start in range(0, len(point_ids), batch_size):
            end = min(start + batch_size, len(point_ids))
            rows = torch.as_tensor(point_ids[start:end], device=store.device)
            captured.clear()
            model(store.batch(rows, normalization))
            if len(captured) != 1:
                raise RuntimeError(f"expected one mass hidden tensor, got {len(captured)}")
            output[start:end] = captured[0].detach()
    finally:
        handle.remove()
    return output


@torch.inference_mode()
def complex_angle_delay(
    channel: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    output = torch.empty(
        (len(channel), e115.PORTS, e115.MH, e115.MV, e115.S_SC),
        dtype=torch.complex64,
        device=channel.device,
    )
    for start in range(0, len(channel), batch_size):
        end = min(start + batch_size, len(channel))
        shaped = (
            channel[start:end]
            .reshape(-1, e115.MP, e115.MH, e115.MV, e115.N_UE, e115.S_SC)
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(-1, e115.PORTS, e115.MH, e115.MV, e115.S_SC)
        )
        output[start:end] = torch.fft.ifft(
            torch.fft.fft2(shaped, dim=(-3, -2)), dim=-1
        )
    return output


def periodic_trilinear_sample(
    field: torch.Tensor,
    coordinate: torch.Tensor,
) -> torch.Tensor:
    """Sample [B,P,H,V,D] at continuous periodic [H,V,D] coordinates."""

    value = field.permute(0, 2, 3, 4, 1)
    lower = torch.floor(coordinate).to(torch.int64)
    fraction = coordinate - lower.to(coordinate.dtype)
    batch = torch.arange(len(field), device=field.device)[:, None]
    output = torch.zeros(
        (*coordinate.shape[:2], field.shape[1]),
        dtype=field.dtype,
        device=field.device,
    )
    periods = (e115.MH, e115.MV, e115.S_SC)
    for dh in (0, 1):
        wh = fraction[..., 0] if dh else 1.0 - fraction[..., 0]
        ih = (lower[..., 0] + dh) % periods[0]
        for dv in (0, 1):
            wv = fraction[..., 1] if dv else 1.0 - fraction[..., 1]
            iv = (lower[..., 1] + dv) % periods[1]
            for dd in (0, 1):
                wd = fraction[..., 2] if dd else 1.0 - fraction[..., 2]
                index_d = (lower[..., 2] + dd) % periods[2]
                weight = (wh * wv * wd)[..., None]
                output = output + weight * value[batch, ih, iv, index_d]
    return output


@torch.inference_mode()
def build_v7_evidence(
    store,
    point_ids: np.ndarray,
    v7_angle_delay: torch.Tensor,
    support: np.ndarray,
) -> tuple[torch.Tensor, list[str]]:
    rows = torch.as_tensor(point_ids, device=store.device)
    coordinate = store.coordinate[rows]
    mask = store.mask[rows]
    mass_logits = store.base_log_mass[rows].masked_fill(~mask, -1e4)
    base_mass = torch.softmax(mass_logits, dim=1)
    base_port = torch.softmax(store.base_log_port[rows], dim=2)
    base_phase_raw = store.base_phase[rows]
    base_phase = torch.complex(
        base_phase_raw[..., : e115.PORTS],
        base_phase_raw[..., e115.PORTS :],
    )
    base_phase = base_phase / base_phase.abs().clamp_min(1e-8)

    coordinate_np = coordinate.cpu().numpy().astype(np.float32)
    mask_np = mask.cpu().numpy()
    patch = e131.patch_energy(support, coordinate_np, mask_np)
    patch_rank = e131.row_ranks(patch, mask_np)
    mass_np = base_mass.cpu().numpy().astype(np.float32)
    mass_rank = e131.row_ranks(mass_np, mask_np)
    patch_tensor = torch.as_tensor(patch, device=store.device)
    patch_rank_tensor = torch.as_tensor(patch_rank, device=store.device)
    mass_rank_tensor = torch.as_tensor(mass_rank, device=store.device)

    sampled = periodic_trilinear_sample(v7_angle_delay, coordinate)
    sampled_power = sampled.abs().square()
    sampled_total = sampled_power.sum(dim=2)
    point_total = v7_angle_delay.abs().square().sum(dim=(1, 2, 3, 4))
    cell_fraction = sampled_total / point_total[:, None].clamp_min(TINY)
    cell_port = sampled_power / sampled_total[..., None].clamp_min(TINY)
    port_overlap_weight = torch.sqrt(base_port.clamp_min(0.0) * cell_port.clamp_min(0.0))
    port_overlap = port_overlap_weight.sum(dim=2)
    sampled_phase = sampled / sampled.abs().clamp_min(1e-8)
    phase_projection = (
        port_overlap_weight * base_phase * torch.conj(sampled_phase)
    ).sum(dim=2) / port_overlap.clamp_min(1e-8)

    evidence = torch.stack(
        (
            torch.log(patch_tensor.clamp_min(1e-8)),
            patch_rank_tensor,
            torch.log(base_mass.clamp_min(1e-8)),
            mass_rank_tensor,
            patch_rank_tensor - mass_rank_tensor,
            torch.log(cell_fraction.clamp_min(1e-10)),
            port_overlap,
            phase_projection.real,
            phase_projection.imag,
        ),
        dim=2,
    ).float()
    evidence = torch.where(mask[..., None], evidence, torch.zeros_like(evidence))
    names = [
        "log_v7_patch_energy",
        "v7_patch_rank",
        "log_base_mass",
        "base_mass_rank",
        "v7_minus_base_rank",
        "log_v7_complex_cell_energy",
        "v7_candidate_port_overlap",
        "v7_candidate_phase_real",
        "v7_candidate_phase_imag",
    ]
    return evidence, names


def normalize_evidence(
    evidence: torch.Tensor,
    mask: torch.Tensor,
    fit_rows: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.as_tensor(fit_rows, device=evidence.device)
    values = evidence[rows][mask[rows]]
    mean = values.mean(dim=0)
    std = values.std(dim=0)
    std = torch.where(std < 1e-4, torch.ones_like(std), std)
    normalized = ((evidence - mean) / std).clamp(-8.0, 8.0)
    normalized = torch.where(mask[..., None], normalized, torch.zeros_like(normalized))
    return normalized, mean, std


@torch.inference_mode()
def render_raw_channels(
    store,
    point_ids: np.ndarray,
    batch_size: int,
) -> torch.Tensor:
    output = torch.empty(
        (len(point_ids), e115.MP * e115.MH * e115.MV, e115.N_UE, e115.S_SC),
        dtype=torch.complex64,
        device=store.device,
    )
    for start in range(0, len(point_ids), batch_size):
        end = min(start + batch_size, len(point_ids))
        rows = torch.as_tensor(point_ids[start:end], device=store.device)
        output[start:end] = e115.render_complex_channel(
            e115.raw_output(raw_batch(store, rows))
        )
    return output


def aggregate_torch_metrics(
    values: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
) -> dict[str, torch.Tensor]:
    pas, pdp, error, energy = values
    pas_mean = pas.mean(dtype=torch.float64)
    pdp_mean = pdp.mean(dtype=torch.float64)
    nmse = error.sum(dtype=torch.float64) / energy.sum(dtype=torch.float64).clamp_min(TINY)
    nmse_score = 1.0 / (1.0 + nmse)
    score = 0.4 * pas_mean + 0.4 * pdp_mean + 0.2 * nmse_score
    return {
        "pas": pas_mean,
        "pdp": pdp_mean,
        "nmse": nmse,
        "nmse_score": nmse_score,
        "score": score,
    }


def candidate_prediction(
    head: V7InLoopMassResidual,
    local_rows: torch.Tensor,
    query_ids_tensor: torch.Tensor,
    store,
    frozen_hidden: torch.Tensor,
    evidence: torch.Tensor,
    raw_channel: torch.Tensor,
    v7: torch.Tensor,
    active: torch.Tensor,
    delta_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    point_ids = query_ids_tensor[local_rows]
    batch = raw_batch(store, point_ids)
    raw = e115.raw_output(batch)
    if delta_override is None:
        delta = head(
            frozen_hidden[local_rows],
            evidence[local_rows],
            batch["mask"],
            raw["mass"],
        )
    else:
        delta = delta_override
    delta = torch.where(active[local_rows, None], delta, torch.zeros_like(delta))
    mass_logits = (raw["mass_logits"] + delta).masked_fill(~batch["mask"], -1e4)
    candidate = dict(raw)
    candidate["mass_logits"] = mass_logits
    candidate["mass"] = torch.softmax(mass_logits, dim=1)
    rendered = e115.render_complex_channel(candidate)
    prediction = v7[local_rows] + rendered - raw_channel[local_rows]
    prediction = torch.where(
        active[local_rows, None, None, None], prediction, v7[local_rows]
    )
    return prediction, delta


def training_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    delta: torch.Tensor,
    mask: torch.Tensor,
    baseline_pas: torch.Tensor,
    baseline_pdp: torch.Tensor,
    baseline_error: torch.Tensor,
    baseline_energy: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    summary = aggregate_torch_metrics(e115.torch_official_metrics(prediction, truth))
    base_pas = baseline_pas.mean(dtype=torch.float64)
    base_pdp = baseline_pdp.mean(dtype=torch.float64)
    base_nmse = baseline_error.sum(dtype=torch.float64) / baseline_energy.sum(
        dtype=torch.float64
    ).clamp_min(TINY)
    barrier = (
        F.relu(base_pas - summary["pas"])
        + F.relu(base_pdp - summary["pdp"])
        + 0.2 * F.relu(summary["nmse"] - base_nmse)
    )
    trust = delta[mask].square().mean() if mask.any() else delta.square().mean()
    loss = (
        -summary["score"]
        + config.component_barrier_weight * barrier
        + config.trust_weight * trust.double()
    )
    return loss, {
        "score": float(summary["score"].detach().cpu()),
        "pas": float(summary["pas"].detach().cpu()),
        "pdp": float(summary["pdp"].detach().cpu()),
        "nmse": float(summary["nmse"].detach().cpu()),
        "barrier": float(barrier.detach().cpu()),
        "trust": float(trust.detach().cpu()),
    }


@torch.inference_mode()
def evaluate_rows(
    head: V7InLoopMassResidual,
    rows: np.ndarray,
    query_ids_tensor: torch.Tensor,
    store,
    frozen_hidden: torch.Tensor,
    evidence: torch.Tensor,
    raw_channel: torch.Tensor,
    v7: torch.Tensor,
    truth: torch.Tensor,
    active: torch.Tensor,
    batch_size: int,
    delta_override: torch.Tensor | None = None,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    arrays = tuple(np.empty(len(rows), dtype=np.float64) for _ in range(4))
    deltas = np.empty((len(rows), store.mask.shape[1]), dtype=np.float32)
    relative = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        local = torch.as_tensor(rows[start:end], device=store.device)
        override = None
        if delta_override is not None:
            override = delta_override[start:end]
        prediction, delta = candidate_prediction(
            head,
            local,
            query_ids_tensor,
            store,
            frozen_hidden,
            evidence,
            raw_channel,
            v7,
            active,
            override,
        )
        values = e115.torch_official_metrics(prediction, truth[local])
        for destination, value in zip(arrays, values, strict=True):
            destination[start:end] = value.cpu().numpy()
        deltas[start:end] = delta.cpu().numpy().astype(np.float32)
        residual_norm = (prediction - v7[local]).abs().square().sum(
            dim=(1, 2, 3), dtype=torch.float64
        ).sqrt()
        base_norm = v7[local].abs().square().sum(
            dim=(1, 2, 3), dtype=torch.float64
        ).sqrt()
        relative[start:end] = (
            residual_norm / base_norm.clamp_min(TINY)
        ).cpu().numpy().astype(np.float32)
    return arrays, deltas, relative


def component_feasible(
    selected: dict[str, float | int],
    baseline: dict[str, float | int],
) -> bool:
    tolerance = 1e-7
    return bool(
        float(selected["pas"]) >= float(baseline["pas"]) - tolerance
        and float(selected["pdp"]) >= float(baseline["pdp"]) - tolerance
        and float(selected["nmse"]) <= float(baseline["nmse"]) + tolerance
    )


def train_fold(
    outer: int,
    config: Config,
    query_ids: np.ndarray,
    query_fold: np.ndarray,
    query_ids_tensor: torch.Tensor,
    v7: torch.Tensor,
    truth: torch.Tensor,
    v7_angle_delay: torch.Tensor,
    support: np.ndarray,
    baseline_values: tuple[np.ndarray, ...],
    active: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, object], tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    inner = (outer + 1) % 5
    fit_rows = np.flatnonzero((query_fold != outer) & (query_fold != inner))
    validation_rows = np.flatnonzero(query_fold == inner)
    held_rows = np.flatnonzero(query_fold == outer)
    model, store, normalization, source_checkpoint = e148.load_fold_model(outer, device)
    frozen_hidden = extract_mass_hidden(
        model, store, normalization, query_ids, config.feature_batch_size
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    evidence_raw, evidence_names = build_v7_evidence(
        store, query_ids, v7_angle_delay, support
    )
    all_ids = torch.as_tensor(query_ids, device=device)
    mask = store.mask[all_ids]
    evidence, evidence_mean, evidence_std = normalize_evidence(
        evidence_raw, mask, fit_rows
    )
    raw_channel = render_raw_channels(store, query_ids, config.feature_batch_size)
    head = V7InLoopMassResidual(evidence.shape[-1], config).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1), eta_min=config.learning_rate * 0.05
    )

    validation_baseline = statistics(
        *(value[validation_rows] for value in baseline_values)
    )
    initial_values, _, _ = evaluate_rows(
        head,
        validation_rows,
        query_ids_tensor,
        store,
        frozen_hidden,
        evidence,
        raw_channel,
        v7,
        truth,
        active,
        config.metric_batch_size,
    )
    initial_summary = statistics(*initial_values)
    best_score = float(initial_summary["score"])
    best_epoch = 0
    best_state = copy.deepcopy(head.state_dict())
    history: list[dict[str, object]] = [
        {
            "epoch": 0,
            "validation": initial_summary,
            "validation_delta": float(
                initial_summary["score"] - validation_baseline["score"]
            ),
            "component_feasible": True,
        }
    ]
    rng = np.random.default_rng((config.seed + outer * 1009) % (2**32))
    baseline_pas = torch.as_tensor(baseline_values[0], device=device)
    baseline_pdp = torch.as_tensor(baseline_values[1], device=device)
    baseline_error = torch.as_tensor(baseline_values[2], device=device)
    baseline_energy = torch.as_tensor(baseline_values[3], device=device)

    for epoch in range(1, config.epochs + 1):
        head.train()
        order = rng.permutation(fit_rows)
        loss_rows: list[float] = []
        score_rows: list[float] = []
        for start in range(0, len(order), config.batch_size):
            batch_np = order[start : start + config.batch_size]
            local = torch.as_tensor(batch_np, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction, delta = candidate_prediction(
                head,
                local,
                query_ids_tensor,
                store,
                frozen_hidden,
                evidence,
                raw_channel,
                v7,
                active,
            )
            point_ids = query_ids_tensor[local]
            loss, diagnostics = training_loss(
                prediction,
                truth[local],
                delta,
                store.mask[point_ids],
                baseline_pas[local],
                baseline_pdp[local],
                baseline_error[local],
                baseline_energy[local],
                config,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss outer={outer} epoch={epoch}")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                head.parameters(), config.gradient_clip
            )
            if not torch.isfinite(gradient):
                raise RuntimeError(f"non-finite gradient outer={outer} epoch={epoch}")
            optimizer.step()
            loss_rows.append(float(loss.detach().cpu()))
            score_rows.append(diagnostics["score"])
        scheduler.step()
        row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": float(np.mean(loss_rows)),
            "train_batch_score": float(np.mean(score_rows)),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        if epoch == 1 or epoch % config.validation_interval == 0 or epoch == config.epochs:
            head.eval()
            validation_values, _, validation_relative = evaluate_rows(
                head,
                validation_rows,
                query_ids_tensor,
                store,
                frozen_hidden,
                evidence,
                raw_channel,
                v7,
                truth,
                active,
                config.metric_batch_size,
            )
            selected = statistics(*validation_values)
            feasible = component_feasible(selected, validation_baseline)
            row.update(
                {
                    "validation": selected,
                    "validation_delta": float(
                        selected["score"] - validation_baseline["score"]
                    ),
                    "validation_relative_residual_median": float(
                        np.median(validation_relative)
                    ),
                    "component_feasible": feasible,
                }
            )
            score = float(selected["score"])
            if feasible and score > best_score + 1e-10:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(head.state_dict())
        history.append(row)
        if epoch == 1 or epoch % 2 == 0 or epoch == config.epochs:
            message = (
                f"E149 outer={outer} epoch={epoch}/{config.epochs} "
                f"train={row['train_batch_score']:.6f}"
            )
            if "validation_delta" in row:
                message += f" inner_delta={row['validation_delta']:+.6f}"
            print(message, flush=True)

    head.load_state_dict(best_state)
    head.eval()
    held_values, held_delta, held_relative = evaluate_rows(
        head,
        held_rows,
        query_ids_tensor,
        store,
        frozen_hidden,
        evidence,
        raw_channel,
        v7,
        truth,
        active,
        config.metric_batch_size,
    )
    held_baseline = statistics(*(value[held_rows] for value in baseline_values))
    held_summary = statistics(*held_values)

    setup, positions, _, site_all, _ = load_positions()
    held_positions = positions[query_ids[held_rows]]
    held_site = site_all[query_ids[held_rows]].astype(np.int8)
    permutation, control = e148.mismatch_permutation(
        held_positions,
        held_site,
        setup.bs_positions,
        config.seed + outer * 2027,
        config.control_distance_strata,
    )
    shuffled_delta = torch.as_tensor(held_delta[permutation], device=device)
    shuffled_values, _, _ = evaluate_rows(
        head,
        held_rows,
        query_ids_tensor,
        store,
        frozen_hidden,
        evidence,
        raw_channel,
        v7,
        truth,
        active,
        config.metric_batch_size,
        shuffled_delta,
    )
    shuffled_summary = statistics(*shuffled_values)

    checkpoint = ARTIFACT_DIR / f"e149_outer{outer}_mass_residual.pt"
    torch.save(
        {
            "experiment": VERSION,
            "outer": outer,
            "inner": inner,
            "config": asdict(config),
            "state_dict": {key: value.cpu() for key, value in best_state.items()},
            "evidence_names": evidence_names,
            "evidence_mean": evidence_mean.cpu(),
            "evidence_std": evidence_std.cpu(),
            "best_epoch": best_epoch,
            "source_path_checkpoint": PATHS.relative(e148.checkpoint_path(outer)),
            "official_h_only": True,
            "sionna_used": False,
            "v12_used": False,
        },
        checkpoint,
    )
    record = {
        "outer": outer,
        "inner": inner,
        "fit_folds": [fold for fold in range(5) if fold not in (outer, inner)],
        "fit_count": int(len(fit_rows)),
        "validation_count": int(len(validation_rows)),
        "held_count": int(len(held_rows)),
        "best_epoch": int(best_epoch),
        "source_path_checkpoint": PATHS.relative(e148.checkpoint_path(outer)),
        "checkpoint": PATHS.relative(checkpoint),
        "history": history,
        "held_baseline": held_baseline,
        "held_selected": held_summary,
        "held_delta": float(held_summary["score"] - held_baseline["score"]),
        "held_relative_residual_median": float(np.median(held_relative)),
        "mismatch_control": {**control, "metrics": shuffled_summary},
        "geometry_margin": float(held_summary["score"] - shuffled_summary["score"]),
    }
    print(
        f"E149 outer={outer} best_epoch={best_epoch} "
        f"held_delta={record['held_delta']:+.6f} "
        f"geometry_margin={record['geometry_margin']:+.6f}",
        flush=True,
    )
    del store, frozen_hidden, evidence_raw, evidence, raw_channel, head
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record, held_values, shuffled_values, held_delta, held_relative


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        feature_batch_size=args.feature_batch_size,
        metric_batch_size=args.metric_batch_size,
    )
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed % (2**32 - 1))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    required = (
        e148.V7_CHANNEL,
        TRAIN_CHANNEL,
        e119.OOF_META,
        e115.FULL_FOLD,
        e131.V7_SUPPORT,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = np.load(e119.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    query_fold = np.load(e115.FULL_FOLD).astype(np.int8)[query_ids]
    v7_np = np.load(e148.V7_CHANNEL, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    truth_np = np.asarray(truth_all[query_ids], dtype=np.complex64)
    baseline_values = e148.metric_arrays(
        np.asarray(v7_np, dtype=np.complex64),
        truth_np,
        device,
        config.metric_batch_size,
    )
    baseline = statistics(*baseline_values)
    expected_v7 = 0.644456345600551
    if abs(float(baseline["score"]) - expected_v7) > 1e-9:
        raise RuntimeError(f"V7 baseline mismatch: {baseline['score']} != {expected_v7}")

    v7 = torch.tensor(np.asarray(v7_np, dtype=np.complex64), device=device)
    truth = torch.tensor(truth_np, device=device)
    active = v7.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64) > TINY
    v7_angle_delay = complex_angle_delay(v7, config.feature_batch_size)
    support = np.load(e131.V7_SUPPORT, mmap_mode="r")
    query_ids_tensor = torch.as_tensor(query_ids, device=device)

    selected_values = tuple(np.empty(len(query_ids), dtype=np.float64) for _ in range(4))
    shuffled_values = tuple(np.empty(len(query_ids), dtype=np.float64) for _ in range(4))
    mass_delta = np.zeros((len(query_ids), 64), dtype=np.float32)
    relative_residual = np.zeros(len(query_ids), dtype=np.float32)
    fold_records = []
    for outer in range(5):
        held_rows = np.flatnonzero(query_fold == outer)
        record, values, control_values, delta, relative = train_fold(
            outer,
            config,
            query_ids,
            query_fold,
            query_ids_tensor,
            v7,
            truth,
            v7_angle_delay,
            support,
            baseline_values,
            active,
            device,
        )
        for destination, value in zip(selected_values, values, strict=True):
            destination[held_rows] = value
        for destination, value in zip(shuffled_values, control_values, strict=True):
            destination[held_rows] = value
        mass_delta[held_rows] = delta
        relative_residual[held_rows] = relative
        fold_records.append(record)

    selected = statistics(*selected_values)
    shuffled = statistics(*shuffled_values)
    fold_delta = [float(record["held_delta"]) for record in fold_records]
    geometry_margin = float(selected["score"] - shuffled["score"])
    gate = {
        "required_score_gain": config.required_score_gain,
        "required_positive_folds": config.required_positive_folds,
        "minimum_geometry_margin": config.minimum_geometry_margin,
        "maximum_worst_fold_loss": config.maximum_worst_fold_loss,
        "score_gain": float(selected["score"] - baseline["score"]),
        "positive_folds": int(sum(value > 0.0 for value in fold_delta)),
        "worst_fold_delta": float(min(fold_delta)),
        "geometry_margin": geometry_margin,
        "pas_non_worse": bool(selected["pas"] >= baseline["pas"]),
        "pdp_non_worse": bool(selected["pdp"] >= baseline["pdp"]),
        "nmse_non_worse": bool(selected["nmse"] <= baseline["nmse"]),
    }
    gate["passed"] = bool(
        gate["score_gain"] >= gate["required_score_gain"]
        and gate["positive_folds"] >= gate["required_positive_folds"]
        and gate["worst_fold_delta"] >= -gate["maximum_worst_fold_loss"]
        and gate["geometry_margin"] >= gate["minimum_geometry_margin"]
        and gate["pas_non_worse"]
        and gate["pdp_non_worse"]
        and gate["nmse_non_worse"]
    )

    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        query_fold=query_fold,
        baseline_pas=baseline_values[0],
        baseline_pdp=baseline_values[1],
        baseline_error=baseline_values[2],
        baseline_energy=baseline_values[3],
        selected_pas=selected_values[0],
        selected_pdp=selected_values[1],
        selected_error=selected_values[2],
        selected_energy=selected_values[3],
        shuffled_pas=shuffled_values[0],
        shuffled_pdp=shuffled_values[1],
        shuffled_error=shuffled_values[2],
        shuffled_energy=shuffled_values[3],
        mass_logit_residual=mass_delta.astype(np.float16),
        relative_channel_residual=relative_residual,
        selected_epoch=np.asarray(
            [record["best_epoch"] for record in fold_records], dtype=np.int16
        ),
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_only_positive" if gate["passed"] else "analysis_only_negative",
        "final_goal": (
            "Official-input-only Round 2 model above V12 local 0.677521 and online 0.70"
        ),
        "single_experimental_goal": (
            "Test whether a path-mass neural residual trained with V7 inside the full "
            "official-score loop can improve V7 without changing path coordinates, ports, "
            "phase, or point-power heads."
        ),
        "primary_hypothesis": (
            "The E120 path representation and V7 candidate-local energy/port/phase evidence "
            "contain useful correction information, but the correction direction must be "
            "learned from zero against V7's actual residual rather than copied from the "
            "absolute path renderer."
        ),
        "protocol": {
            "canonical_fixed500": True,
            "outer_folds": 5,
            "inner_rule": "(outer + 1) % 5",
            "path_backbone_excludes_outer_and_inner": True,
            "residual_head_fit_folds_only": True,
            "inner_selects_epoch": True,
            "outer_final_score_only": True,
            "strict_zero_initialization": True,
            "v7_exact_skip_connection": True,
            "learned_state": "bounded path log-mass residual only",
            "frozen_states": ["coordinate", "port", "phase", "point_power"],
            "official_h_only": True,
            "sionna_used": False,
            "v12_used": False,
            "submission_generated": False,
        },
        "config": asdict(config),
        "baseline_v7": baseline,
        "selected": {
            "metrics": selected,
            "delta_to_v7": float(selected["score"] - baseline["score"]),
            "fold_delta": fold_delta,
            "relative_channel_residual_median": float(
                np.median(relative_residual)
            ),
        },
        "mismatch_control": shuffled,
        "geometry_margin": geometry_margin,
        "folds": fold_records,
        "gate": gate,
        "decision": {
            "passed": bool(gate["passed"]),
            "next_if_passed": (
                "Rebuild the same V7-in-loop mass residual on all 4000 official points, "
                "then add coordinate state only in a separate gated experiment."
            ),
            "next_if_failed": (
                "Stop mass-only residual scaling; the next physical-network experiment must "
                "change the path representation or jointly model a different state rather "
                "than tuning epochs, width, or residual strength."
            ),
        },
        "outputs": {
            "metrics": PATHS.relative(METRICS),
            "report": PATHS.relative(REPORT),
            "checkpoints": [record["checkpoint"] for record in fold_records],
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "baseline_v7": baseline,
                "selected": report["selected"],
                "mismatch_control": shuffled,
                "gate": gate,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
