#!/usr/bin/env python3
"""Train nested, state-specific path operators without overwriting a learned core."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import train_e115_complete_path_state_operator as e115
from project_paths import PATHS


VERSION = "Round2-E116-nested-state-specific-path-operator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "nested_path_state_operator"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_nested_path_state_operator"
SCREEN_REPORT = ANALYSIS_DIR / "nested_path_state_outer0_screen.json"
STATE_NAMES = ("mass", "coordinate", "port", "phase")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--core-epochs", type=int, default=24)
    parser.add_argument("--scene-epochs", type=int, default=20)
    parser.add_argument("--graph-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    return parser.parse_args()


def positional_features(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    row = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    column = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(row, column, indexing="ij")
    return torch.stack(
        (
            yy,
            xx,
            torch.sin(math.pi * yy),
            torch.cos(math.pi * yy),
            torch.sin(math.pi * xx),
            torch.cos(math.pi * xx),
        ),
        dim=2,
    ).reshape(height * width, 6)


class PositionalSceneEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.route = e115.convolutional_scene_branch(11, hidden, (4, 12))
        self.panorama = e115.convolutional_scene_branch(9, hidden, (2, 16))
        self.polar = e115.convolutional_scene_branch(4, hidden, (4, 16))
        self.corridor = e115.convolutional_scene_branch(4, hidden, (4, 16))
        self.position = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.type_embedding = nn.Parameter(torch.zeros(4, hidden))
        nn.init.normal_(self.type_embedding, std=0.02)

    def tokens(self, value: torch.Tensor, type_index: int) -> torch.Tensor:
        batch, hidden, height, width = value.shape
        position = self.position(
            positional_features(height, width, value.device, value.dtype)
        )
        return (
            value.flatten(2).transpose(1, 2)
            + position[None]
            + self.type_embedding[type_index][None, None]
        ).reshape(batch, height * width, hidden)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            (
                self.tokens(self.route(batch["route"]), 0),
                self.tokens(self.panorama(batch["panorama"]), 1),
                self.tokens(self.polar(batch["polar"]), 2),
                self.tokens(self.corridor(batch["corridor"]), 3),
            ),
            dim=1,
        )


class NestedPathStateOperator(e115.CompletePathStateOperator):
    def __init__(
        self,
        candidate_dim: int,
        context_dim: int,
        graph_dim: int,
        config: e115.Config,
        variant: e115.Variant,
    ) -> None:
        super().__init__(candidate_dim, context_dim, graph_dim, config, variant)
        hidden = config.hidden
        self.scene_encoder = PositionalSceneEncoder(hidden)
        self.scene_context_norm = nn.LayerNorm(hidden)
        self.graph_context_norm = nn.LayerNorm(hidden)
        self.scene_gates = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(hidden)) for name in STATE_NAMES}
        )
        self.graph_gates = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(hidden)) for name in STATE_NAMES}
        )

    def state_hidden(
        self,
        core: torch.Tensor,
        state: str,
        scene: torch.Tensor | None,
        graph: torch.Tensor | None,
    ) -> torch.Tensor:
        output = core
        if scene is not None:
            output = output + torch.tanh(self.scene_gates[state])[None, None] * scene
        if graph is not None:
            output = output + torch.tanh(self.graph_gates[state])[None, None] * graph
        return output

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch["mask"]
        hidden = self.candidate_input(batch["candidate_feature"])
        context = self.context_input(batch["context"])
        hidden = hidden + context[:, None]
        hidden = self.candidate_encoder(hidden, src_key_padding_mask=~mask)
        core = self.output_norm(hidden)

        scene_context = None
        if self.variant.use_scene:
            scene = self.scene_encoder(batch)
            attended, _ = self.scene_attention(core, scene, scene, need_weights=False)
            scene_context = self.scene_context_norm(attended)

        graph_context = None
        if self.variant.use_graph:
            batch_size, candidates, neighbors, _ = batch["graph_feature"].shape
            graph = self.graph_input(batch["graph_feature"])
            graph = graph.reshape(
                batch_size * candidates, neighbors, self.config.hidden
            )
            graph = self.graph_encoder(graph)
            graph = graph.mean(dim=1).reshape(
                batch_size, candidates, self.config.hidden
            )
            graph_context = self.graph_context_norm(graph)

        mass_hidden = self.state_hidden(core, "mass", scene_context, graph_context)
        coordinate_hidden = self.state_hidden(
            core, "coordinate", scene_context, graph_context
        )
        port_hidden = self.state_hidden(core, "port", scene_context, graph_context)
        phase_hidden = self.state_hidden(core, "phase", scene_context, graph_context)

        score_residual = self.config.score_limit * torch.tanh(
            self.score_head(mass_hidden)[..., 0]
        )
        mass_logits = batch["base_log_mass"] + score_residual
        mass_logits = mass_logits.masked_fill(~mask, -1e4)
        mass = torch.softmax(mass_logits, dim=1)
        coordinate_scale = torch.as_tensor(
            self.config.coordinate_limits,
            device=core.device,
            dtype=core.dtype,
        )
        coordinate_residual = (
            torch.tanh(self.coordinate_head(coordinate_hidden)) * coordinate_scale
        )
        coordinate = batch["coordinate"] + coordinate_residual
        port_residual = self.config.port_limit * torch.tanh(self.port_head(port_hidden))
        port = torch.softmax(batch["base_log_port"] + port_residual, dim=2)
        base_complex = torch.complex(
            batch["base_phase"][..., : e115.PORTS],
            batch["base_phase"][..., e115.PORTS :],
        )
        base_angle = torch.angle(base_complex)
        phase_residual = self.config.phase_limit * torch.tanh(
            self.phase_head(phase_hidden)
        )
        return {
            "mass": mass,
            "mass_logits": mass_logits,
            "coordinate": coordinate,
            "coordinate_residual": coordinate_residual,
            "port": port,
            "port_residual": port_residual,
            "phase": base_angle + phase_residual,
            "phase_residual": phase_residual,
            "score_residual": score_residual,
            "zero_logit": torch.zeros(len(core), device=core.device, dtype=core.dtype),
            "log_power": batch["base_log_power"],
        }


def stage_prefixes(stage: str) -> tuple[str, ...]:
    if stage == "scene":
        return (
            "scene_encoder.",
            "scene_attention.",
            "scene_context_norm.",
            "scene_gates.",
        )
    if stage == "graph":
        return (
            "graph_input.",
            "graph_encoder.",
            "graph_context_norm.",
            "graph_gates.",
        )
    return ()


def configure_stage(model: NestedPathStateOperator, stage: str) -> list[nn.Parameter]:
    scene_prefixes = stage_prefixes("scene")
    graph_prefixes = stage_prefixes("graph")
    selected: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        is_scene = name.startswith(scene_prefixes)
        is_graph = name.startswith(graph_prefixes)
        is_unused_point_head = name.startswith(("zero_head.", "power_head."))
        is_replaced_fusion = name.startswith(("scene_norm.", "graph_fusion."))
        if stage == "core":
            trainable = (
                not is_scene
                and not is_graph
                and not is_unused_point_head
                and not is_replaced_fusion
            )
        else:
            trainable = name.startswith(stage_prefixes(stage))
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
    model.eval()
    if stage == "core":
        for module in (
            model.candidate_input,
            model.context_input,
            model.candidate_encoder,
            model.output_norm,
            model.score_head,
            model.coordinate_head,
            model.port_head,
            model.phase_head,
        ):
            module.train()
    elif stage == "scene":
        model.scene_encoder.train()
        model.scene_attention.train()
        model.scene_context_norm.train()
    else:
        model.graph_input.train()
        model.graph_encoder.train()
        model.graph_context_norm.train()
    return selected


def gate_summary(model: NestedPathStateOperator) -> dict[str, dict[str, float]]:
    return {
        branch: {
            state: float(torch.tanh(gates[state]).abs().mean().detach().cpu().item())
            for state in STATE_NAMES
        }
        for branch, gates in (
            ("scene", model.scene_gates),
            ("graph", model.graph_gates),
        )
    }


def train_stage(
    model: NestedPathStateOperator,
    store: e115.TensorStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    held_rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    config: e115.Config,
    stage: str,
    variant: e115.Variant,
    seed: int,
) -> dict[str, object]:
    effective_seed = seed % (2**32)
    torch.manual_seed(effective_seed)
    np.random.seed(effective_seed)
    model.variant = variant
    trainable = configure_stage(model, stage)
    learning_rate = config.learning_rate if stage == "core" else 3e-4
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1), eta_min=learning_rate * 0.05
    )
    initial = e115.evaluate_rows(
        model,
        store,
        validation_rows,
        normalization,
        config,
        config.batch_size,
        False,
    )
    best_validation = float(initial["path_endpoint_mean"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = [
        {
            "epoch": 0.0,
            "train_loss": float("nan"),
            "validation_path_endpoint": best_validation,
            "validation_mass_cosine": float(initial["mass_hellinger_cosine"]),
        }
    ]
    rng = np.random.default_rng(effective_seed)
    for epoch in range(1, config.epochs + 1):
        configure_stage(model, stage)
        order = rng.permutation(fit_rows)
        losses = []
        for start in range(0, len(order), config.batch_size):
            rows = torch.as_tensor(
                order[start : start + config.batch_size], device=store.device
            )
            batch = store.batch(rows, normalization)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, _ = e115.state_objective(output, batch, config)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss stage={stage} epoch={epoch}")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 2.0)
            if not torch.isfinite(gradient):
                raise RuntimeError(f"non-finite gradient stage={stage} epoch={epoch}")
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        row = {"epoch": float(epoch), "train_loss": float(np.mean(losses))}
        if epoch % config.validation_interval == 0:
            validation = e115.evaluate_rows(
                model,
                store,
                validation_rows,
                normalization,
                config,
                config.batch_size,
                False,
            )
            score = float(validation["path_endpoint_mean"])
            row["validation_path_endpoint"] = score
            row["validation_mass_cosine"] = float(
                validation["mass_hellinger_cosine"]
            )
            if score > best_validation:
                best_validation = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(row)
        if epoch % 4 == 0 or epoch == config.epochs:
            message = (
                f"stage={stage} epoch={epoch}/{config.epochs} "
                f"train={row['train_loss']:.6f}"
            )
            if "validation_path_endpoint" in row:
                message += f" validation={row['validation_path_endpoint']:.6f}"
            print(message, flush=True)

    model.load_state_dict(best_state)
    validation = e115.evaluate_rows(
        model,
        store,
        validation_rows,
        normalization,
        config,
        config.batch_size,
        True,
    )
    held = e115.evaluate_rows(
        model,
        store,
        held_rows,
        normalization,
        config,
        config.batch_size,
        True,
    )
    checkpoint = ARTIFACT_DIR / f"outer0_nested_{stage}.pt"
    torch.save(
        {
            "experiment": VERSION,
            "stage": stage,
            "variant": asdict(variant),
            "config": asdict(config),
            "effective_seed": effective_seed,
            "state_dict": model.state_dict(),
            "normalization": {key: value.cpu() for key, value in normalization.items()},
            "best_epoch": best_epoch,
            "official_h_only": True,
            "sionna_used": False,
            "v12_power_anchored": True,
        },
        checkpoint,
    )
    return {
        "stage": stage,
        "variant": asdict(variant),
        "effective_seed": effective_seed,
        "trainable_parameter_count": int(sum(value.numel() for value in trainable)),
        "best_epoch": int(best_epoch),
        "history": history,
        "validation": e115.compact_metrics(validation),
        "held": e115.compact_metrics(held),
        "gates": gate_summary(model),
        "checkpoint": PATHS.relative(checkpoint),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    base_config = e115.Config(
        epochs=args.core_epochs,
        batch_size=args.batch_size,
        zero_weight=0.0,
        power_weight=0.0,
    )
    store = e115.TensorStore(e115.cache_dir("screen_block01"), device)
    fold_id = store.fold_id.cpu().numpy()
    held_rows = np.flatnonzero(fold_id == 0)
    validation_rows = np.flatnonzero(fold_id == 1)
    fit_rows = np.flatnonzero(np.isin(fold_id, (2, 3, 4)))
    normalization = store.normalization(fit_rows)
    baseline_validation = e115.evaluate_rows(
        None,
        store,
        validation_rows,
        normalization,
        base_config,
        args.batch_size,
        True,
    )
    baseline_held = e115.evaluate_rows(
        None,
        store,
        held_rows,
        normalization,
        base_config,
        args.batch_size,
        True,
    )

    seed = base_config.random_state % (2**32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = NestedPathStateOperator(
        store.candidate_feature.shape[-1],
        store.context.shape[-1],
        store.graph_feature.shape[-1],
        base_config,
        e115.Variant("candidate_core", False, False),
    ).to(device)
    stages = []
    stages.append(
        train_stage(
            model,
            store,
            fit_rows,
            validation_rows,
            held_rows,
            normalization,
            base_config,
            "core",
            e115.Variant("candidate_core", False, False),
            seed,
        )
    )
    stages.append(
        train_stage(
            model,
            store,
            fit_rows,
            validation_rows,
            held_rows,
            normalization,
            replace(base_config, epochs=args.scene_epochs),
            "scene",
            e115.Variant("candidate_scene", True, False),
            seed + 1,
        )
    )
    stages.append(
        train_stage(
            model,
            store,
            fit_rows,
            validation_rows,
            held_rows,
            normalization,
            replace(base_config, epochs=args.graph_epochs),
            "graph",
            e115.Variant("complete", True, True),
            seed + 2,
        )
    )
    core = stages[0]
    complete = stages[-1]
    ranking = sorted(
        stages,
        key=lambda row: float(row["validation"]["path_endpoint_mean"]),
        reverse=True,
    )
    gate = {
        "minimum_held_gain_over_raw": 0.03,
        "minimum_complete_gain_over_core": 0.003,
        "held_gain_over_raw": float(
            complete["held"]["path_endpoint_mean"]
            - baseline_held["path_endpoint_mean"]
        ),
        "complete_gain_over_core": float(
            complete["held"]["path_endpoint_mean"]
            - core["held"]["path_endpoint_mean"]
        ),
        "validation_selected_stage": ranking[0]["stage"],
    }
    gate["passed"] = bool(
        gate["held_gain_over_raw"] >= gate["minimum_held_gain_over_raw"]
        and gate["complete_gain_over_core"]
        >= gate["minimum_complete_gain_over_core"]
        and gate["validation_selected_stage"] == "graph"
    )
    report = {
        "experiment": VERSION,
        "status": "outer0_screen_complete",
        "primary_hypothesis": (
            "State-specific, position-aware scene and query-continuity residuals add "
            "path information when they are nested on a frozen candidate core, while "
            "anchoring V12 power prevents the dominant E115 negative-transfer mode."
        ),
        "official_h_only": True,
        "sionna_used": False,
        "protocol": {
            "globally_blocked_folds": [0, 1],
            "fit_folds": [2, 3, 4],
            "validation_fold": 1,
            "held_fold": 0,
            "nested_frozen_branch_training": True,
            "held_labels_seen": False,
            "held_h_used_as_donor_anywhere": False,
            "validation_h_used_as_donor_anywhere": False,
        },
        "base_config": asdict(base_config),
        "baseline": {
            "validation": e115.compact_metrics(baseline_validation),
            "held": e115.compact_metrics(baseline_held),
        },
        "stages": stages,
        "validation_ranking": [row["stage"] for row in ranking],
        "deployment_gate": gate,
        "seconds": float(time.time() - started),
    }
    SCREEN_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "baseline": report["baseline"],
                "stages": [
                    {
                        "stage": row["stage"],
                        "best_epoch": row["best_epoch"],
                        "validation": row["validation"],
                        "held": row["held"],
                        "gates": row["gates"],
                    }
                    for row in stages
                ],
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
