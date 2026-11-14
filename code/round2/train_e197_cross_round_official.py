#!/usr/bin/env python3
"""Pretrain E190 on Round 1 official H, then fine-tune strict Round 2 folds."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any

import numpy as np
import torch

import prepare_e197_cross_round_official as prepare
import train_e190_official_sionna_path_calibrator as e190
import train_e195_directional_probe_calibrator as e195
import train_surface_interaction_energy_model as e85
from common.official_metrics import aggregate_statistics, point_metric_statistics_torch
from project_paths import PATHS
from run_rbf_spectral_field import statistics


VERSION = "Round2-E197-cross-round-official-path-calibrator"
ARTIFACT_DIR = prepare.ARTIFACT_DIR
FOLD_DIR = ARTIFACT_DIR / "folds"
ANALYSIS_DIR = prepare.ANALYSIS_DIR
LOG_DIR = ANALYSIS_DIR / "logs"
REPORT = ANALYSIS_DIR / "fixed500_report.json"
OUTPUT_CHANNEL = ARTIFACT_DIR / "fixed500_channel.npy"
PRETRAIN_CHECKPOINT = ARTIFACT_DIR / "round1_pretrain.pt"
V15_REPORT = e195.V15_REPORT

POINTS_R2 = 4000
CANDIDATES = e85.PATHS_PER_POINT
PORTS = e85.MP * e85.N_UE
PERIODS = np.asarray((e85.MH, e85.MV, e85.S_SC), dtype=np.float32)
V15_SCORE = 0.6863750036074728
TARGET_SCORE = 0.700000
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    outer: int = 0
    validation_fold: int = 1
    pretrain_epochs: int = 32
    finetune_epochs: int = 48
    validation_interval: int = 4
    batch_size: int = 64
    endpoint_batch_size: int = 32
    hidden: int = 96
    heads: int = 4
    sequence_layers: int = 2
    set_layers: int = 2
    feedforward: int = 256
    dropout: float = 0.08
    learning_rate: float = 5e-4
    weight_decay: float = 2e-4
    score_limit: float = 4.0
    port_limit: float = 2.0
    random_state: int = 20260813197


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("fold", "aggregate", "profile"), default="fold"
    )
    parser.add_argument("--outer", type=int, choices=range(5), default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--force-fold", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def row_digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def fold_paths(outer: int) -> dict[str, Path]:
    return {
        "pretrain": PRETRAIN_CHECKPOINT,
        "screen": FOLD_DIR / f"outer{outer}_screen.pt",
        "refit": FOLD_DIR / f"outer{outer}_refit.pt",
        "channel": FOLD_DIR / f"outer{outer}_channel.npy",
        "endpoint": FOLD_DIR / f"outer{outer}_endpoint.npz",
        "report": ANALYSIS_DIR / f"outer{outer}_report.json",
        "failure": ANALYSIS_DIR / f"outer{outer}_failure.json",
    }


def required_inputs() -> tuple[Path, ...]:
    return (
        *prepare.expected_files(),
        e195.INDEPENDENT_BASE,
        e195.INDEPENDENT_BASE_REPORT,
        e195.TRAIN_CHANNEL,
        e195.FOLD_CACHE,
        V15_REPORT,
        *tuple(
            e195.fold_paths(outer)[name]
            for outer in range(5)
            for name in ("mapping", "screen_labels", "refit_labels")
        ),
    )


def model_config(config: Config) -> e85.Config:
    return e85.Config(
        hidden=config.hidden,
        heads=config.heads,
        sequence_layers=config.sequence_layers,
        set_layers=config.set_layers,
        feedforward=config.feedforward,
        dropout=config.dropout,
        epochs=config.finetune_epochs,
        minimum_epochs=0,
        validation_interval=config.validation_interval,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        score_limit=config.score_limit,
        port_limit=config.port_limit,
        random_state=config.random_state,
    )


def objective_config(config: Config) -> e190.Config:
    return e190.Config(
        outer=config.outer,
        validation_fold=config.validation_fold,
        epochs=config.finetune_epochs,
        minimum_epochs=0,
        validation_interval=config.validation_interval,
        batch_size=config.batch_size,
        hidden=config.hidden,
        heads=config.heads,
        sequence_layers=config.sequence_layers,
        set_layers=config.set_layers,
        feedforward=config.feedforward,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        score_limit=config.score_limit,
        port_limit=config.port_limit,
        random_state=config.random_state,
    )


class Round1Store:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.base = torch.as_tensor(np.load(prepare.BASE), device=device)
        self.fraction = torch.as_tensor(np.load(prepare.FRACTION), device=device)
        self.presence = torch.as_tensor(np.load(prepare.PRESENCE), device=device)
        self.coordinate = torch.as_tensor(np.load(prepare.COORDINATE), device=device)
        self.port = torch.as_tensor(
            np.load(prepare.PORT).astype(np.float32).reshape(prepare.POINTS, CANDIDATES, PORTS),
            device=device,
        )
        self.sequence_geometry = torch.as_tensor(
            np.load(prepare.SEQUENCE_GEOMETRY), device=device
        )
        self.sequence_hash = torch.as_tensor(
            np.load(prepare.SEQUENCE_HASH).astype(np.int64), device=device
        )
        self.depth = torch.as_tensor(
            np.load(prepare.DEPTH).astype(np.int64), device=device
        )
        self.interactions = torch.as_tensor(
            np.load(prepare.INTERACTIONS).astype(np.int64), device=device
        )
        self.objects = torch.as_tensor(
            np.load(prepare.OBJECTS).astype(np.int64), device=device
        )
        self.primitives = torch.as_tensor(
            np.load(prepare.PRIMITIVES).astype(np.int64), device=device
        )
        self.target_pas = torch.as_tensor(np.load(prepare.TARGET_PAS), device=device)
        self.target_pdp = torch.as_tensor(np.load(prepare.TARGET_PDP), device=device)
        self.power = np.load(prepare.POINT_POWER).astype(np.float64)
        self.mask = torch.as_tensor(np.load(prepare.MASK), device=device)
        self.labels = e195.ResidentLabels(prepare.LABELS, device)

    def batch(
        self, rows: np.ndarray, mean: torch.Tensor, std: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        ids = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        anchored = self.fraction[ids].clamp_min(1e-12) * torch.sqrt(
            self.presence[ids].clamp_min(1.0) / 24.0
        )
        batch = {
            "base": (self.base[ids] - mean) / std,
            "base_log_mass": torch.log(anchored.clamp_min(1e-12)),
            "coordinate": self.coordinate[ids],
            "base_log_port": torch.log(self.port[ids].clamp_min(1e-8)),
            "sequence_geometry": self.sequence_geometry[ids].float(),
            "sequence_hash": self.sequence_hash[ids],
            "depth": self.depth[ids],
            "interactions": self.interactions[ids],
            "objects": self.objects[ids],
            "primitives": self.primitives[ids],
        }
        return batch, self.target_pas[ids], self.target_pdp[ids]

    def normalization(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.base.mean(dim=(0, 1))
        variance = self.base.square().mean(dim=(0, 1)) - mean.square()
        std = variance.clamp_min(1e-8).sqrt()
        std = torch.where(std < 1e-4, torch.ones_like(std), std)
        return mean, std


def make_model(config: Config, device: torch.device, seed: int) -> e85.SurfaceInteractionModel:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sequence_dim = int(np.load(e85.SEQUENCE_GEOMETRY, mmap_mode="r").shape[-1])
    return e85.SurfaceInteractionModel(120, sequence_dim, model_config(config), "geometry").to(
        device
    )


def train_epoch(
    model: e85.SurfaceInteractionModel,
    optimizer: torch.optim.Optimizer,
    store: Round1Store | e195.ResidentStore,
    labels: e195.ResidentLabels,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    config: Config,
    rng: np.random.Generator,
    round1: bool,
) -> dict[str, float]:
    model.train()
    eligible = rows[labels.valid[rows] & (store.power[rows] > 0.0)]
    order = rng.permutation(eligible)
    totals = {
        name: []
        for name in (
            "loss",
            "pas",
            "pdp",
            "mass_affinity",
            "coordinate",
            "port",
            "existence",
        )
    }
    objective = objective_config(config)
    for start in range(0, len(order), config.batch_size):
        local = order[start : start + config.batch_size]
        ids = torch.as_tensor(local, dtype=torch.long, device=store.device)
        if round1:
            batch, target_pas, target_pdp = store.batch(local, mean, std)
        else:
            batch, target_pas, target_pdp = store.batch(
                local, mean, std, include_probe=False
            )
        output = model(batch)
        loss, values = e190.label_objective(
            output,
            batch,
            target_pas,
            target_pdp,
            labels.mass[ids],
            labels.delta[ids],
            labels.port[ids],
            objective,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError("non-finite E197 gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        totals["loss"].append(float(loss.detach().item()))
        for name, value in values.items():
            totals[name].append(float(value.detach().item()))
    return {name: float(np.mean(values)) for name, values in totals.items()}


def pretrain(
    path: Path, store: Round1Store, config: Config, device: torch.device
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_file():
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if int(checkpoint.get("epochs", -1)) == config.pretrain_epochs:
            return checkpoint, {
                "status": "reused",
                "path": PATHS.relative(path),
                "epochs": config.pretrain_epochs,
            }
    started = time.time()
    model = make_model(config, device, config.random_state)
    mean, std = store.normalization()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.pretrain_epochs,
        eta_min=config.learning_rate * 0.08,
    )
    rows = np.arange(prepare.POINTS, dtype=np.int64)
    rng = np.random.default_rng(config.random_state)
    history = []
    for epoch in range(1, config.pretrain_epochs + 1):
        values = train_epoch(
            model,
            optimizer,
            store,
            store.labels,
            rows,
            mean,
            std,
            config,
            rng,
            True,
        )
        scheduler.step()
        history.append({"epoch": epoch, **values})
        if epoch % config.validation_interval == 0 or epoch == config.pretrain_epochs:
            print(
                f"E197 outer={config.outer} Round1 pretrain "
                f"epoch={epoch}/{config.pretrain_epochs} "
                f"mass={values['mass_affinity']:.6f} "
                f"pas={values['pas']:.6f} pdp={values['pdp']:.6f}",
                flush=True,
            )
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "feature_mean": mean.cpu().numpy(),
        "feature_std": std.cpu().numpy(),
        "epochs": config.pretrain_epochs,
        "history": history,
    }
    atomic_torch_save(path, checkpoint)
    return checkpoint, {
        "status": "created",
        "path": PATHS.relative(path),
        "epochs": config.pretrain_epochs,
        "seconds": float(time.time() - started),
    }


def load_pretrained_model(
    checkpoint: dict[str, Any], config: Config, device: torch.device, seed: int
) -> e85.SurfaceInteractionModel:
    model = make_model(config, device, seed)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def endpoint_arrays(
    model: e85.SurfaceInteractionModel,
    store: e195.ResidentStore,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    mass = np.empty((len(rows), CANDIDATES), dtype=np.float32)
    coordinate = np.empty((len(rows), CANDIDATES, 3), dtype=np.float32)
    port = np.empty((len(rows), CANDIDATES, e85.MP, e85.N_UE), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch, _, _ = store.batch(
                rows[start:end], mean, std, include_probe=False
            )
            output = model(batch)
            mass[start:end] = output["mass"].cpu().numpy().astype(np.float32)
            coordinate[start:end] = (
                output["coordinate"].cpu().numpy() % PERIODS
            ).astype(np.float32)
            port[start:end] = output["port"].cpu().numpy().astype(np.float32)
    return mass, coordinate, port


def evaluate_model(
    model: e85.SurfaceInteractionModel,
    store: e195.ResidentStore,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    evaluator: e195.IndependentEvaluator,
    config: Config,
    collect_channel: bool = False,
) -> tuple[dict[str, float | int], dict[str, np.ndarray], np.ndarray | None]:
    mass, coordinate, port = endpoint_arrays(
        model, store, rows, mean, std, config.endpoint_batch_size
    )
    pas, pdp = e190.render_endpoint(mass, coordinate, port, evaluator.device)
    metrics, channel = evaluator.evaluate(rows, pas, pdp, collect_channel=collect_channel)
    return metrics, {"mass": mass, "coordinate": coordinate, "port": port}, channel


def screen_finetune(
    checkpoint: dict[str, Any],
    store: e195.ResidentStore,
    labels: e195.ResidentLabels,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    evaluator: e195.IndependentEvaluator,
    config: Config,
    device: torch.device,
) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
    model = load_pretrained_model(
        checkpoint, config, device, config.random_state + config.outer
    )
    baseline, _, _ = evaluate_model(
        model, store, validation_rows, mean, std, evaluator, config
    )
    best_score = float(baseline["score"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, Any]] = [
        {"epoch": 0, "validation": baseline, "source": "Round1 pretrained"}
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.finetune_epochs,
        eta_min=config.learning_rate * 0.08,
    )
    rng = np.random.default_rng(config.random_state + config.outer)
    for epoch in range(1, config.finetune_epochs + 1):
        values = train_epoch(
            model,
            optimizer,
            store,
            labels,
            fit_rows,
            mean,
            std,
            config,
            rng,
            False,
        )
        scheduler.step()
        record: dict[str, Any] = {"epoch": epoch, "train": values}
        if epoch % config.validation_interval == 0:
            validation, _, _ = evaluate_model(
                model, store, validation_rows, mean, std, evaluator, config
            )
            record["validation"] = validation
            record["delta_vs_pretrained"] = e190.metric_delta(validation, baseline)
            if float(validation["score"]) > best_score:
                best_score = float(validation["score"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"E197 outer={config.outer} finetune "
                f"epoch={epoch}/{config.finetune_epochs} "
                f"validation={float(validation['score']):.6f} best={best_score:.6f}",
                flush=True,
            )
        history.append(record)
    return best_state, best_epoch, history


def refit(
    checkpoint: dict[str, Any],
    store: e195.ResidentStore,
    labels: e195.ResidentLabels,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    epochs: int,
    config: Config,
    device: torch.device,
) -> tuple[e85.SurfaceInteractionModel, list[dict[str, float]]]:
    model = load_pretrained_model(
        checkpoint, config, device, config.random_state + 100_000 + config.outer
    )
    if epochs == 0:
        return model, []
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=config.learning_rate * 0.08
    )
    rng = np.random.default_rng(config.random_state + 100_000 + config.outer)
    history = []
    for epoch in range(1, epochs + 1):
        values = train_epoch(
            model,
            optimizer,
            store,
            labels,
            rows,
            mean,
            std,
            config,
            rng,
            False,
        )
        scheduler.step()
        history.append({"epoch": epoch, **values})
        if epoch % config.validation_interval == 0 or epoch == epochs:
            print(
                f"E197 refit outer={config.outer} epoch={epoch}/{epochs} "
                f"mass={values['mass_affinity']:.6f} "
                f"pas={values['pas']:.6f} pdp={values['pdp']:.6f}",
                flush=True,
            )
    return model, history


def round2_mode(
    store: e195.ResidentStore, outer: int, mode: str, device: torch.device
) -> tuple[e195.ResidentLabels, np.ndarray, dict[str, Any]]:
    if mode not in ("screen", "refit"):
        raise ValueError(mode)
    source_paths = e195.fold_paths(outer)
    mapping = np.load(source_paths["mapping"], allow_pickle=False)
    latent = mapping[f"{mode}_latent"].astype(np.float32)
    coordinate = e195.candidate_coordinates(latent, device)
    store.set_mode(coordinate, np.empty(0, dtype=np.float16))
    label_path = source_paths[f"{mode}_labels"]
    labels = e195.ResidentLabels(label_path, device)
    return labels, coordinate, {
        "mapping": PATHS.relative(source_paths["mapping"]),
        "labels": PATHS.relative(label_path),
        "latent_digest": row_digest(latent),
        "coordinate_digest": row_digest(coordinate),
    }


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    device = torch.device(args.device)
    config = Config(outer=args.outer, validation_fold=(args.outer + 1) % 5)
    r1 = Round1Store(device)
    checkpoint, pretrain_report = pretrain(fold_paths(args.outer)["pretrain"], r1, config, device)
    store = e195.ResidentStore(device)
    evaluator = e195.IndependentEvaluator(device)
    split = e195.canonical_split(config.outer, config.validation_fold, evaluator)
    labels, _, lineage = round2_mode(store, config.outer, "screen", device)
    mean, std = store.normalization(split["screen_fit"])
    model = load_pretrained_model(checkpoint, config, device, config.random_state)
    rows = split["screen_fit"][
        labels.valid[split["screen_fit"]] & (store.power[split["screen_fit"]] > 0.0)
    ][: config.batch_size]
    if len(rows) == 0:
        raise RuntimeError("E197 profile found no eligible Round2 rows")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    values = train_epoch(
        model,
        optimizer,
        store,
        labels,
        rows,
        mean,
        std,
        config,
        np.random.default_rng(config.random_state),
        False,
    )
    return {
        "experiment": VERSION,
        "status": "profile_complete",
        "pretrain": pretrain_report,
        "lineage": lineage,
        "split": {key: int(len(value)) if isinstance(value, np.ndarray) else value for key, value in split.items()},
        "one_batch_gradient": values,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "peak_torch_memory_gib": float(torch.cuda.max_memory_allocated(device) / 2**30)
        if device.type == "cuda"
        else 0.0,
        "seconds": float(time.time() - started),
    }


def run_fold(args: argparse.Namespace) -> dict[str, Any]:
    paths = fold_paths(args.outer)
    if paths["report"].is_file() and not args.force_fold:
        return json.loads(paths["report"].read_text(encoding="utf-8"))
    started = time.time()
    phase: dict[str, float] = {}
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    config = Config(outer=args.outer, validation_fold=(args.outer + 1) % 5)
    lineage = e195.validate_lineage()
    evaluator = e195.IndependentEvaluator(device)
    split = e195.canonical_split(config.outer, config.validation_fold, evaluator)

    phase_start = time.time()
    r1 = Round1Store(device)
    pretrain_checkpoint, pretrain_report = pretrain(
        paths["pretrain"], r1, config, device
    )
    phase["round1_pretrain"] = float(time.time() - phase_start)
    del r1
    if device.type == "cuda":
        torch.cuda.empty_cache()

    store = e195.ResidentStore(device)
    phase_start = time.time()
    screen_labels, _, screen_lineage = round2_mode(
        store, config.outer, "screen", device
    )
    mean_screen, std_screen = store.normalization(split["screen_fit"])
    best_state, best_epoch, screen_history = screen_finetune(
        pretrain_checkpoint,
        store,
        screen_labels,
        split["screen_fit"],
        split["validation"],
        mean_screen,
        std_screen,
        evaluator,
        config,
        device,
    )
    atomic_torch_save(
        paths["screen"],
        {
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "config": asdict(config),
            "history": screen_history,
            "split_digest": row_digest(
                split["screen_fit"], split["validation"], split["outer"]
            ),
        },
    )
    phase["round2_screen"] = float(time.time() - phase_start)
    del screen_labels
    if device.type == "cuda":
        torch.cuda.empty_cache()

    phase_start = time.time()
    refit_labels, _, refit_lineage = round2_mode(store, config.outer, "refit", device)
    mean_refit, std_refit = store.normalization(split["refit"])
    model, refit_history = refit(
        pretrain_checkpoint,
        store,
        refit_labels,
        split["refit"],
        mean_refit,
        std_refit,
        best_epoch,
        config,
        device,
    )
    selected, endpoint, channel = evaluate_model(
        model,
        store,
        split["outer"],
        mean_refit,
        std_refit,
        evaluator,
        config,
        collect_channel=True,
    )
    if channel is None:
        raise RuntimeError("E197 outer channel missing")
    phase["round2_refit_and_outer"] = float(time.time() - phase_start)

    v15 = json.loads(V15_REPORT.read_text(encoding="utf-8"))
    v15_fold = next(row for row in v15["folds"] if int(row["outer"]) == config.outer)
    delta = e190.metric_delta(selected, v15_fold["selected"])
    atomic_save_npy(paths["channel"], channel)
    atomic_save_npz(
        paths["endpoint"],
        query_ids=split["outer"].astype(np.int32),
        mass=endpoint["mass"],
        coordinate=endpoint["coordinate"],
        port=endpoint["port"],
    )
    atomic_torch_save(
        paths["refit"],
        {
            "state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "config": asdict(config),
            "feature_mean": mean_refit.cpu().numpy(),
            "feature_std": std_refit.cpu().numpy(),
            "history": refit_history,
            "split_digest": row_digest(split["refit"], split["outer"]),
        },
    )
    phase["save"] = float(time.time() - started - sum(phase.values()))

    report = {
        "experiment": VERSION,
        "status": "fold_complete",
        "configuration": asdict(config),
        "outer": config.outer,
        "validation_fold": config.validation_fold,
        "split": {
            "screen_fit": int(len(split["screen_fit"])),
            "refit": int(len(split["refit"])),
            "validation": int(len(split["validation"])),
            "outer": int(len(split["outer"])),
            "zero_label_overlap": split["leakage"],
        },
        "pretrain": pretrain_report,
        "selection": {
            "best_finetune_epoch": best_epoch,
            "screen_history": screen_history,
            "refit_history": refit_history,
        },
        "selected": selected,
        "v15": v15_fold["selected"],
        "delta_vs_v15": delta,
        "data_lineage": {
            "round1_official_train_h_used": True,
            "round1_test_h_used": False,
            "round2_outer_h_used": False,
            "historical_v7_v15_prediction_h_read": False,
            "historical_test_endpoint_read": False,
            "round1_cache": PATHS.relative(prepare.R1_DIR),
            "round2_screen": screen_lineage,
            "round2_refit": refit_lineage,
            "independent_base_lineage": lineage.get("data_lineage", {}),
        },
        "output": {
            "channel": PATHS.relative(paths["channel"]),
            "endpoint": PATHS.relative(paths["endpoint"]),
            "checkpoint": PATHS.relative(paths["refit"]),
            "shape": list(channel.shape),
            "dtype": str(channel.dtype),
            "finite": bool(np.isfinite(channel).all()),
        },
        "performance": {
            "device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
            "peak_torch_memory_gib": float(
                torch.cuda.max_memory_allocated(device) / 2**30
            )
            if device.type == "cuda"
            else 0.0,
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "timing_seconds": phase,
            "measured_bottleneck": max(phase, key=phase.get),
        },
        "seconds": float(time.time() - started),
    }
    atomic_json(paths["report"], report)
    paths["failure"].unlink(missing_ok=True)
    return report


def aggregate(device: torch.device) -> dict[str, Any]:
    started = time.time()
    e195.validate_lineage()
    evaluator = e195.IndependentEvaluator(device)
    output = np.empty((500, 256, 4, 192), dtype=np.complex64)
    fold_reports = []
    deltas = []
    for outer in range(5):
        paths = fold_paths(outer)
        if not paths["report"].is_file() or not paths["channel"].is_file():
            raise FileNotFoundError(f"missing E197 outer {outer}")
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        rows = evaluator.fold_id == outer
        local = np.load(paths["channel"], mmap_mode="r")
        expected = (int(rows.sum()), 256, 4, 192)
        if local.shape != expected or local.dtype != np.complex64:
            raise ValueError(f"invalid E197 outer{outer}: {local.shape}/{local.dtype}")
        output[rows] = np.asarray(local, dtype=np.complex64)
        fold_reports.append(
            {
                "outer": outer,
                "count": int(rows.sum()),
                "selected": report["selected"],
                "v15": report["v15"],
                "delta_vs_v15": report["delta_vs_v15"],
                "best_finetune_epoch": report["selection"]["best_finetune_epoch"],
                "performance": report["performance"],
            }
        )
        deltas.append(float(report["delta_vs_v15"]["score"]))
    selected = evaluator.score_channel(output)
    v15 = json.loads(V15_REPORT.read_text(encoding="utf-8"))["selected"]
    delta = e190.metric_delta(selected, v15)
    independent = point_metric_statistics_torch(
        torch.as_tensor(output, device=device), evaluator.truth
    )
    replay = aggregate_statistics(independent)
    replay_error = max(abs(float(selected[name]) - replay[name]) for name in replay)
    if replay_error > 1e-7:
        raise RuntimeError(f"E197 official replay mismatch: {replay_error}")
    positive_folds = int(sum(value > 0.0 for value in deltas))
    worst_fold = float(min(deltas))
    passed = bool(
        float(selected["score"]) >= TARGET_SCORE
        and positive_folds >= 4
        and worst_fold >= -0.001
    )
    atomic_save_npy(OUTPUT_CHANNEL, output)
    report = {
        "experiment": VERSION,
        "status": "fixed500_passed" if passed else "fixed500_failed",
        "selected": selected,
        "v15": v15,
        "delta_vs_v15": delta,
        "delta_vs_target": float(selected["score"] - TARGET_SCORE),
        "folds": fold_reports,
        "gate": {
            "target_score": TARGET_SCORE,
            "minimum_positive_folds": 4,
            "minimum_worst_fold_delta": -0.001,
            "positive_folds": positive_folds,
            "worst_fold_delta": worst_fold,
            "passed": passed,
        },
        "official_replay": {**replay, "max_abs_error": replay_error},
        "output": {
            "channel": PATHS.relative(OUTPUT_CHANNEL),
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(np.isfinite(output).all()),
        },
        "seconds": float(time.time() - started),
    }
    atomic_json(REPORT, report)
    return report


def main() -> None:
    args = parse_args()
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    missing = [PATHS.relative(path) for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing E197 inputs: {missing}")
    try:
        if args.stage == "profile":
            report = run_profile(args)
        elif args.stage == "fold":
            report = run_fold(args)
        else:
            report = aggregate(torch.device(args.device))
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        if args.stage == "fold":
            failure = {
                "experiment": VERSION,
                "status": "failed",
                "outer": args.outer,
                "traceback": traceback.format_exc(),
            }
            atomic_json(fold_paths(args.outer)["failure"], failure)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
