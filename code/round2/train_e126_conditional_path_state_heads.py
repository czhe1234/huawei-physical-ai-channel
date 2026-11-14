#!/usr/bin/env python3
"""Train conditional path-state heads over the frozen E125 identity core."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e123_complete_path_variable_system as e123
import analyze_e124_one_to_one_path_identity_ceiling as e124
import train_e115_complete_path_state_operator as e115
import train_e118_hierarchical_unique_path_operator as e118
from common.official_metrics import PointMetricStatistics, aggregate_statistics
from project_paths import PATHS


VERSION = "Round2-E126-conditional-path-state-heads"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "conditional_path_state_heads"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_conditional_path_state_heads"
REPORT = ANALYSIS_DIR / "conditional_path_state_outer0_screen.json"
IDENTITY_CHECKPOINT = (
    PATHS.artifacts
    / "round2"
    / "identity_first_path_operator"
    / "outer0_identity_core.pt"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
OUTER = 0
INNER = 1
STAGES = ("coordinate", "port", "common_phase", "relative_phase", "power")
PERIODS = torch.tensor((e115.MH, e115.MV, e115.S_SC), dtype=torch.float32)
COORDINATE_LIMITS = torch.tensor((1.5, 1.0, 3.0), dtype=torch.float32)
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--coordinate-epochs", type=int, default=24)
    parser.add_argument("--port-epochs", type=int, default=20)
    parser.add_argument("--common-phase-epochs", type=int, default=24)
    parser.add_argument("--relative-phase-epochs", type=int, default=24)
    parser.add_argument("--power-epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=96)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--validation-interval", type=int, default=2)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Write isolated smoke checkpoints/report instead of the formal screen outputs.",
    )
    parser.add_argument("--seed", type=int, default=20_260_808_126)
    return parser.parse_args()


def unit_complex(value: torch.Tensor) -> torch.Tensor:
    return value / value.abs().clamp_min(1e-12)


def periodic_features(coordinate: torch.Tensor) -> torch.Tensor:
    period = PERIODS.to(coordinate.device, coordinate.dtype)
    angle = 2.0 * math.pi * coordinate / period
    return torch.cat((torch.sin(angle), torch.cos(angle)), dim=2)


def periodic_delta(target: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    period = PERIODS.to(target.device, target.dtype)
    return torch.remainder(target - base + period / 2.0, period) - period / 2.0


class ConditionalStateHeads(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.coordinate = nn.Sequential(
            nn.Linear(hidden + 3, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.port = nn.Sequential(
            nn.Linear(hidden + 3 + 6, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, e115.PORTS),
        )
        phase_input = hidden + 3 + 6 + e115.PORTS + 2 * e115.PORTS
        self.common_phase = nn.Sequential(
            nn.Linear(phase_input, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.relative_phase = nn.Sequential(
            nn.Linear(phase_input + 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, e115.PORTS),
        )
        self.power = nn.Sequential(
            nn.Linear(hidden + 1 + 1 + 6, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        for module in (
            self.coordinate,
            self.port,
            self.common_phase,
            self.relative_phase,
            self.power,
        ):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)

    @staticmethod
    def mass_features(mass: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (mass, torch.sqrt(mass.clamp_min(1e-12)), torch.log(mass.clamp_min(1e-8))),
            dim=2,
        )

    def predict_coordinate(
        self,
        core: torch.Tensor,
        mass: torch.Tensor,
        base_coordinate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = torch.cat((core, self.mass_features(mass)), dim=2)
        limit = COORDINATE_LIMITS.to(core.device, core.dtype)
        delta = torch.tanh(self.coordinate(feature)) * limit
        period = PERIODS.to(core.device, core.dtype)
        return torch.remainder(base_coordinate + delta, period), delta

    def predict_port(
        self,
        core: torch.Tensor,
        mass: torch.Tensor,
        coordinate: torch.Tensor,
        base_log_port: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = torch.cat(
            (core, self.mass_features(mass), periodic_features(coordinate)), dim=2
        )
        residual = 2.5 * torch.tanh(self.port(feature))
        return torch.softmax(base_log_port + residual, dim=2), residual

    def phase_features(
        self,
        core: torch.Tensor,
        mass: torch.Tensor,
        coordinate: torch.Tensor,
        port: torch.Tensor,
        base_phase: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                core,
                self.mass_features(mass),
                periodic_features(coordinate),
                torch.sqrt(port.clamp_min(1e-12)),
                base_phase.real,
                base_phase.imag,
            ),
            dim=2,
        )

    def predict_common_phase(
        self,
        core: torch.Tensor,
        mass: torch.Tensor,
        coordinate: torch.Tensor,
        port: torch.Tensor,
        base_phase: torch.Tensor,
    ) -> torch.Tensor:
        feature = self.phase_features(core, mass, coordinate, port, base_phase)
        return math.pi * torch.tanh(self.common_phase(feature)[..., 0])

    def predict_relative_phase(
        self,
        core: torch.Tensor,
        mass: torch.Tensor,
        coordinate: torch.Tensor,
        port: torch.Tensor,
        base_phase: torch.Tensor,
        common: torch.Tensor,
    ) -> torch.Tensor:
        feature = self.phase_features(core, mass, coordinate, port, base_phase)
        feature = torch.cat(
            (feature, torch.sin(common)[..., None], torch.cos(common)[..., None]),
            dim=2,
        )
        residual = math.pi * torch.tanh(self.relative_phase(feature))
        anchor = port.argmax(dim=2, keepdim=True)
        anchor_phase = torch.gather(residual, 2, anchor)
        residual = residual - anchor_phase
        return torch.atan2(torch.sin(residual), torch.cos(residual))

    def predict_power(
        self,
        core: torch.Tensor,
        mass: torch.Tensor,
        coordinate: torch.Tensor,
        base_log_power: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = torch.einsum("bk,bkh->bh", mass, core)
        entropy = -(
            mass * torch.log(mass.clamp_min(1e-8))
        ).sum(dim=1, keepdim=True)
        coordinate_summary = torch.einsum(
            "bk,bkd->bd", mass, periodic_features(coordinate)
        )
        feature = torch.cat(
            (pooled, base_log_power[:, None], entropy, coordinate_summary), dim=1
        )
        residual = 4.0 * torch.tanh(self.power(feature)[:, 0])
        return base_log_power + residual, residual


def load_identity_core(
    store: e118.ClusterTensorStore,
    device: torch.device,
) -> tuple[
    e118.HierarchicalUniquePathOperator,
    dict[str, torch.Tensor],
    dict[str, object],
]:
    checkpoint = torch.load(IDENTITY_CHECKPOINT, map_location="cpu", weights_only=False)
    config = e118.Config(**checkpoint["config"])
    model = e118.HierarchicalUniquePathOperator(
        store.raw.candidate_feature.shape[-1],
        store.cluster_feature.shape[-1],
        store.raw.context.shape[-1],
        store.graph_feature.shape[-1],
        config,
        e115.VARIANTS[0],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    normalization = {
        key: value.to(device) for key, value in checkpoint["normalization"].items()
    }
    return model, normalization, checkpoint


@torch.inference_mode()
def precompute_identity(
    model: e118.HierarchicalUniquePathOperator,
    store: e118.ClusterTensorStore,
    normalization: dict[str, torch.Tensor],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    core_parts: list[torch.Tensor] = []
    mass_parts: list[torch.Tensor] = []
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = model.output_norm.register_forward_hook(hook)
    for start in range(0, len(store.mask), batch_size):
        stop = min(start + batch_size, len(store.mask))
        rows = torch.arange(start, stop, device=store.device)
        batch = store.batch(rows, normalization)
        captured.clear()
        output = model(batch)
        if len(captured) != 1:
            raise RuntimeError("identity core hook did not capture exactly one tensor")
        core_parts.append(captured[0])
        mass_parts.append(output["mass"].detach())
    handle.remove()
    return torch.cat(core_parts), torch.cat(mass_parts)


def hard_targets(
    directory,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    component = np.load(e114.COMPONENTS, allow_pickle=False)
    base_coordinate = np.asarray(
        np.load(directory / "coordinate.npy", mmap_mode="r"), np.float32
    )
    mask = np.asarray(np.load(directory / "mask.npy", mmap_mode="r"), bool)
    target, assignment = e124.build_one_to_one_targets(
        base_coordinate,
        mask,
        component["coordinates"].astype(np.float32),
        component["coefficients"].astype(np.complex64),
        component["point_power"].astype(np.float64),
    )
    mass_absolute = torch.as_tensor(
        target["mass_absolute"], dtype=torch.float32, device=device
    )
    mass_total = mass_absolute.sum(dim=1)
    mass = mass_absolute / mass_total[:, None].clamp_min(TINY)
    coordinate = torch.as_tensor(target["coordinate"], device=device)
    port = torch.as_tensor(target["port"], device=device)
    phase = torch.as_tensor(target["phase"], device=device)
    power = torch.as_tensor(
        component["point_power"].astype(np.float64), device=device
    )
    return (
        {
            "mass_absolute": mass_absolute,
            "mass": mass,
            "active": mass_absolute > 0.0,
            "coordinate": coordinate,
            "port": port,
            "phase": phase,
            "power": power,
        },
        assignment,
    )


def base_phase(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return unit_complex(
        torch.complex(
            batch["base_phase"][..., : e115.PORTS],
            batch["base_phase"][..., e115.PORTS :],
        )
    )


def target_phase_decomposition(
    base: torch.Tensor,
    target: torch.Tensor,
    port: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = unit_complex(target * torch.conj(base))
    anchor = port.argmax(dim=2, keepdim=True)
    common_complex = torch.gather(delta, 2, anchor)[..., 0]
    common = torch.angle(common_complex)
    relative = torch.angle(delta * torch.conj(common_complex[..., None]))
    return common, relative


def decode(
    heads: ConditionalStateHeads,
    core: torch.Tensor,
    identity_mass: torch.Tensor,
    batch: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    stage: str,
    teacher: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    stage_index = STAGES.index(stage)
    base_coord = batch["coordinate"]
    base_port_value = torch.softmax(batch["base_log_port"], dim=2)
    base_phase_value = base_phase(batch)
    target_common, target_relative = target_phase_decomposition(
        base_phase_value, target["phase"], target["port"]
    )

    mass = target["mass"] if teacher else identity_mass
    predicted_coordinate, coordinate_residual = heads.predict_coordinate(
        core, mass, base_coord
    )
    if teacher and stage_index > 0:
        coordinate = target["coordinate"]
    elif stage_index >= 0:
        coordinate = predicted_coordinate
    else:
        coordinate = base_coord

    predicted_port, port_residual = heads.predict_port(
        core, mass, coordinate, batch["base_log_port"]
    )
    if teacher and stage_index > 1:
        port = target["port"]
    elif stage_index >= 1:
        port = predicted_port
    else:
        port = base_port_value

    predicted_common = heads.predict_common_phase(
        core, mass, coordinate, port, base_phase_value
    )
    if teacher and stage_index > 2:
        common = target_common
    elif stage_index >= 2:
        common = predicted_common
    else:
        common = torch.zeros_like(predicted_common)

    predicted_relative = heads.predict_relative_phase(
        core, mass, coordinate, port, base_phase_value, common
    )
    if teacher and stage_index > 3:
        relative = target_relative
    elif stage_index >= 3:
        relative = predicted_relative
    else:
        relative = torch.zeros_like(predicted_relative)
    phase = torch.angle(base_phase_value) + common[..., None] + relative

    predicted_log_power, power_residual = heads.predict_power(
        core, mass, coordinate, batch["base_log_power"]
    )
    log_power = predicted_log_power if stage_index >= 4 else batch["base_log_power"]
    output = {
        "mass": mass,
        "coordinate": coordinate,
        "port": port,
        "phase": phase,
        "log_power": log_power,
        "zero_logit": torch.zeros(len(core), device=core.device),
    }
    diagnostics = {
        "coordinate": predicted_coordinate,
        "coordinate_residual": coordinate_residual,
        "port": predicted_port,
        "port_residual": port_residual,
        "common": predicted_common,
        "relative": predicted_relative,
        "log_power": predicted_log_power,
        "power_residual": power_residual,
        "target_common": target_common,
        "target_relative": target_relative,
    }
    return output, diagnostics


def stage_loss(
    stage: str,
    diagnostics: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> torch.Tensor:
    active = target["active"]
    weight = target["mass_absolute"]
    weight = weight / weight.sum().clamp_min(TINY)
    if stage == "coordinate":
        target_delta = periodic_delta(target["coordinate"], batch["coordinate"])
        limit = COORDINATE_LIMITS.to(target_delta.device, target_delta.dtype)
        error = F.smooth_l1_loss(
            diagnostics["coordinate_residual"] / limit,
            target_delta / limit,
            reduction="none",
        ).mean(dim=2)
        return (error * weight).sum() + 0.002 * diagnostics[
            "coordinate_residual"
        ].square().mean()
    if stage == "port":
        affinity = (
            torch.sqrt(diagnostics["port"].clamp_min(1e-12))
            * torch.sqrt(target["port"].clamp_min(0.0))
        ).sum(dim=2)
        return ((1.0 - affinity) * weight).sum() + 0.002 * diagnostics[
            "port_residual"
        ].square().mean()
    if stage == "common_phase":
        cosine = torch.cos(diagnostics["common"] - diagnostics["target_common"])
        return ((1.0 - cosine) * weight).sum() + 0.001 * diagnostics[
            "common"
        ].square().mean()
    if stage == "relative_phase":
        cosine = torch.cos(
            diagnostics["relative"] - diagnostics["target_relative"]
        )
        cosine = (cosine * target["port"]).sum(dim=2)
        return ((1.0 - cosine) * weight).sum() + 0.001 * diagnostics[
            "relative"
        ].square().mean()
    if stage == "power":
        valid = target["power"] > 0.0
        target_log = torch.log(target["power"][valid].clamp_min(TINY)).float()
        return F.smooth_l1_loss(diagnostics["log_power"][valid], target_log) + 0.002 * diagnostics[
            "power_residual"
        ].square().mean()
    raise ValueError(stage)


def state_metric(
    stage: str,
    diagnostics: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> tuple[float, str]:
    weight = target["mass_absolute"]
    weight = weight / weight.sum().clamp_min(TINY)
    if stage == "coordinate":
        delta = periodic_delta(diagnostics["coordinate"], target["coordinate"])
        value = (delta.abs().mean(dim=2) * weight).sum()
        return float(value.cpu()), "lower"
    if stage == "port":
        affinity = (
            torch.sqrt(diagnostics["port"].clamp_min(1e-12))
            * torch.sqrt(target["port"].clamp_min(0.0))
        ).sum(dim=2)
        return float((affinity * weight).sum().cpu()), "higher"
    if stage == "common_phase":
        value = torch.cos(
            diagnostics["common"] - diagnostics["target_common"]
        )
        return float((value * weight).sum().cpu()), "higher"
    if stage == "relative_phase":
        value = torch.cos(
            diagnostics["relative"] - diagnostics["target_relative"]
        )
        value = (value * target["port"]).sum(dim=2)
        return float((value * weight).sum().cpu()), "higher"
    if stage == "power":
        valid = target["power"] > 0.0
        error = (
            diagnostics["log_power"][valid]
            - torch.log(target["power"][valid].clamp_min(TINY)).float()
        ).abs()
        return float(error.mean().cpu()), "lower"
    raise ValueError(stage)


@torch.inference_mode()
def evaluate(
    heads: ConditionalStateHeads,
    core_all: torch.Tensor,
    identity_mass_all: torch.Tensor,
    store: e118.ClusterTensorStore,
    rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    target_all: dict[str, torch.Tensor],
    stage: str,
    teacher: bool,
    batch_size: int,
) -> dict[str, object]:
    heads.eval()
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    pas_parts: list[np.ndarray] = []
    pdp_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    energy_parts: list[np.ndarray] = []
    state_values: list[tuple[float, float]] = []
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        index = torch.as_tensor(local, dtype=torch.long, device=store.device)
        batch = store.batch(index, normalization)
        target = {key: value[index] for key, value in target_all.items()}
        output, diagnostics = decode(
            heads,
            core_all[index],
            identity_mass_all[index],
            batch,
            target,
            stage,
            teacher,
        )
        prediction = e115.render_complex_channel(output)
        truth = torch.as_tensor(
            np.asarray(truth_all[local], dtype=np.complex64), device=store.device
        )
        pas, pdp, error, energy = e115.torch_official_metrics(prediction, truth)
        pas_parts.append(pas.cpu().numpy())
        pdp_parts.append(pdp.cpu().numpy())
        error_parts.append(error.cpu().numpy())
        energy_parts.append(energy.cpu().numpy())
        value, direction = state_metric(stage, diagnostics, batch, target)
        state_values.append((value, float(len(local))))
    values = PointMetricStatistics(
        pas=np.concatenate(pas_parts),
        pdp=np.concatenate(pdp_parts),
        squared_error=np.concatenate(error_parts),
        truth_energy=np.concatenate(energy_parts),
    )
    weighted_state = sum(value * count for value, count in state_values) / sum(
        count for _, count in state_values
    )
    return {
        "mode": "teacher_forced" if teacher else "rollout",
        "stage": stage,
        "state_metric": weighted_state,
        "state_metric_direction": direction,
        "full_official": aggregate_statistics(values),
    }


def stage_parameters(heads: ConditionalStateHeads, stage: str) -> list[nn.Parameter]:
    module = getattr(heads, stage)
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    return list(module.parameters())


def train_stage(
    heads: ConditionalStateHeads,
    stage: str,
    epochs: int,
    core_all: torch.Tensor,
    identity_mass_all: torch.Tensor,
    store: e118.ClusterTensorStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    target_all: dict[str, torch.Tensor],
    train_batch_size: int,
    metric_batch_size: int,
    validation_interval: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    parameters = stage_parameters(heads, stage)
    optimizer = torch.optim.AdamW(parameters, lr=8e-4, weight_decay=2e-4)
    initial_teacher = evaluate(
        heads,
        core_all,
        identity_mass_all,
        store,
        validation_rows,
        normalization,
        target_all,
        stage,
        True,
        metric_batch_size,
    )
    initial_rollout = evaluate(
        heads,
        core_all,
        identity_mass_all,
        store,
        validation_rows,
        normalization,
        target_all,
        stage,
        False,
        metric_batch_size,
    )
    best_score = float(initial_rollout["full_official"]["score"])
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone() for key, value in heads.state_dict().items()
    }
    history = [
        {
            "epoch": 0,
            "teacher_validation": initial_teacher,
            "rollout_validation": initial_rollout,
        }
    ]
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        heads.train()
        shuffled = rng.permutation(fit_rows)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(shuffled), train_batch_size):
            local = shuffled[start : start + train_batch_size]
            index = torch.as_tensor(local, dtype=torch.long, device=store.device)
            batch = store.batch(index, normalization)
            target = {key: value[index] for key, value in target_all.items()}
            optimizer.zero_grad(set_to_none=True)
            _, diagnostics = decode(
                heads,
                core_all[index],
                identity_mass_all[index],
                batch,
                target,
                stage,
                True,
            )
            loss = stage_loss(stage, diagnostics, batch, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        if epoch % validation_interval == 0 or epoch == epochs:
            teacher_validation = evaluate(
                heads,
                core_all,
                identity_mass_all,
                store,
                validation_rows,
                normalization,
                target_all,
                stage,
                True,
                metric_batch_size,
            )
            rollout_validation = evaluate(
                heads,
                core_all,
                identity_mass_all,
                store,
                validation_rows,
                normalization,
                target_all,
                stage,
                False,
                metric_batch_size,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / max(batches, 1),
                    "teacher_validation": teacher_validation,
                    "rollout_validation": rollout_validation,
                }
            )
            teacher_score = float(
                teacher_validation["full_official"]["score"]
            )
            score = float(rollout_validation["full_official"]["score"])
            print(
                f"stage={stage} epoch={epoch}/{epochs} "
                f"loss={total_loss / max(batches, 1):.6f} "
                f"teacher_score={teacher_score:.6f} "
                f"rollout_score={score:.6f} "
                f"teacher_state={teacher_validation['state_metric']:.6f} "
                f"rollout_state={rollout_validation['state_metric']:.6f}",
                flush=True,
            )
            if score > best_score + 1e-8:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in heads.state_dict().items()
                }
    heads.load_state_dict(best_state)
    return (
        {
            "stage": stage,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection_metric": "inner rollout full official score",
            "best_rollout_validation_score": best_score,
            "history": history,
        },
        best_state,
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    directory = e118.cache_dir(e123.cache_tag(OUTER))
    store = e118.ClusterTensorStore(directory, device)
    identity_model, normalization, identity_checkpoint = load_identity_core(
        store, device
    )
    core_all, identity_mass_all = precompute_identity(
        identity_model, store, normalization, args.metric_batch_size
    )
    del identity_model
    target_all, assignment = hard_targets(directory, device)
    fold_id = np.load(e115.FULL_FOLD).astype(np.int8)
    fit_rows = np.flatnonzero((fold_id != OUTER) & (fold_id != INNER))
    validation_rows = np.flatnonzero(fold_id == INNER)
    held_rows = np.flatnonzero(fold_id == OUTER)

    heads = ConditionalStateHeads(core_all.shape[-1]).to(device)
    epoch_map = {
        "coordinate": args.coordinate_epochs,
        "port": args.port_epochs,
        "common_phase": args.common_phase_epochs,
        "relative_phase": args.relative_phase_epochs,
        "power": args.power_epochs,
    }
    stage_records = []
    held_curves = []
    for stage_index, stage in enumerate(STAGES):
        record, state = train_stage(
            heads,
            stage,
            epoch_map[stage],
            core_all,
            identity_mass_all,
            store,
            fit_rows,
            validation_rows,
            normalization,
            target_all,
            args.train_batch_size,
            args.metric_batch_size,
            args.validation_interval,
            args.seed + 1009 * stage_index,
        )
        teacher_held = evaluate(
            heads,
            core_all,
            identity_mass_all,
            store,
            held_rows,
            normalization,
            target_all,
            stage,
            True,
            args.metric_batch_size,
        )
        rollout_held = evaluate(
            heads,
            core_all,
            identity_mass_all,
            store,
            held_rows,
            normalization,
            target_all,
            stage,
            False,
            args.metric_batch_size,
        )
        suffix = "_smoke" if args.smoke else ""
        checkpoint = ARTIFACT_DIR / f"outer0_conditional_{stage}{suffix}.pt"
        torch.save(
            {
                "experiment": VERSION,
                "stage": stage,
                "state_dict": state,
                "hidden": int(core_all.shape[-1]),
                "identity_checkpoint": PATHS.relative(IDENTITY_CHECKPOINT),
                "identity_checkpoint_stage": identity_checkpoint["stage"],
                "official_h_only": True,
                "one_to_one_identity_supervision": True,
            },
            checkpoint,
        )
        record["checkpoint"] = PATHS.relative(checkpoint)
        stage_records.append(record)
        held_curves.append(
            {
                "stage": stage,
                "teacher_forced": teacher_held,
                "rollout": rollout_held,
                "teacher_rollout_gap": float(
                    teacher_held["full_official"]["score"]
                    - rollout_held["full_official"]["score"]
                ),
            }
        )
        print(
            f"stage={stage} held_teacher={teacher_held['full_official']['score']:.6f} "
            f"held_rollout={rollout_held['full_official']['score']:.6f}",
            flush=True,
        )

    identity_reference = json.loads(
        (
            PATHS.analysis
            / "results"
            / "round2_identity_first_path_operator"
            / "identity_first_outer0_screen.json"
        ).read_text(encoding="utf-8")
    )["held_results"][0]["metrics"]["full_official"]
    final_rollout = held_curves[-1]["rollout"]["full_official"]
    positive_rollout_stages = sum(
        item["rollout"]["full_official"]["score"]
        > identity_reference["score"]
        for item in held_curves
    )
    gate = {
        "minimum_positive_rollout_stages": 2,
        "minimum_final_rollout_gain_over_identity": 0.005,
        "positive_rollout_stages": int(positive_rollout_stages),
        "final_rollout_gain_over_identity": float(
            final_rollout["score"] - identity_reference["score"]
        ),
    }
    gate["passed"] = bool(
        gate["positive_rollout_stages"] >= gate["minimum_positive_rollout_stages"]
        and gate["final_rollout_gain_over_identity"]
        >= gate["minimum_final_rollout_gain_over_identity"]
    )
    report = {
        "experiment": VERSION,
        "status": "outer0_screen_complete",
        "primary_hypothesis": (
            "A hierarchical teacher-forced decoder can learn each downstream path "
            "state, while rollout curves expose the first layer where identity errors "
            "destroy complete-channel gains."
        ),
        "strict_protocol": {
            "outer": OUTER,
            "inner": INNER,
            "fit_count": int(len(fit_rows)),
            "validation_count": int(len(validation_rows)),
            "held_count": int(len(held_rows)),
            "held_used_for_epoch_selection": False,
            "identity_core_frozen": True,
            "official_h_only": True,
            "sionna_or_v12_used": False,
            "smoke_run": bool(args.smoke),
        },
        "architecture": {
            "identity_hidden": int(core_all.shape[-1]),
            "decoder_order": list(STAGES),
            "coordinate_limits": COORDINATE_LIMITS.tolist(),
            "port_logit_limit": 2.5,
            "phase_limit": math.pi,
            "power_log_residual_limit": 4.0,
            "teacher_forcing": "all states before the current head use one-to-one truth",
            "rollout": "identity and every trained state use model predictions",
        },
        "assignment": assignment,
        "identity_reference": identity_reference,
        "stages": stage_records,
        "held_curves": held_curves,
        "gate": gate,
        "decision": {
            "conditional_decoder_passed": bool(gate["passed"]),
            "next_if_passed": (
                "Run five strict outer models, then add bounded scene-conditioned "
                "identity/lifecycle features only where rollout uncertainty is low."
            ),
            "next_if_failed": (
                "Stop before fivefold deployment and redesign the first failing state "
                "head using its teacher/rollout gap."
            ),
        },
        "seconds": float(time.time() - started),
    }
    report_path = (
        ANALYSIS_DIR / "conditional_path_state_outer0_smoke.json"
        if args.smoke
        else REPORT
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "identity_reference": identity_reference,
                "held_curves": held_curves,
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
