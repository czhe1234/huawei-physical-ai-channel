#!/usr/bin/env python3
"""Audit the complete-score contribution of each V15 physical input group."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np
import torch

import analyze_e86_surface_endpoint_v11_blend as e86
import optimize_e98_global_array_angle_mapping as e98
import train_e190_official_sionna_path_calibrator as e190
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics


V7_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "full_sionna_migration"
    / "oof_v7_nested_channel.npy"
)
EXPERIMENTS = {
    "v11": {
        "version": "Round2-E203-v15-information-contribution",
        "artifact_dir": PATHS.artifacts / "round2" / "e203_v15_information_contribution",
        "analysis_dir": PATHS.analysis
        / "results"
        / "round2_e203_v15_information_contribution",
        "base_channel": e86.V11_CHANNEL,
    },
    "v7": {
        "version": "Round2-E204-v15-information-contribution-v7-base",
        "artifact_dir": PATHS.artifacts
        / "round2"
        / "e204_v15_information_contribution_v7_base",
        "analysis_dir": PATHS.analysis
        / "results"
        / "round2_e204_v15_information_contribution_v7_base",
        "base_channel": V7_CHANNEL,
    },
}


def configure_experiment(base_name: str) -> None:
    global BASE_NAME, VERSION, ARTIFACT_DIR, ANALYSIS_DIR, FINAL_REPORT, BASE_CHANNEL
    if base_name not in EXPERIMENTS:
        raise ValueError(f"unknown base {base_name!r}")
    spec = EXPERIMENTS[base_name]
    BASE_NAME = base_name
    VERSION = str(spec["version"])
    ARTIFACT_DIR = Path(spec["artifact_dir"])
    ANALYSIS_DIR = Path(spec["analysis_dir"])
    FINAL_REPORT = ANALYSIS_DIR / "v15_information_contribution.json"
    BASE_CHANNEL = Path(spec["base_channel"])


configure_experiment("v11")

VARIANTS = (
    "full",
    "no_angle_context",
    "no_delay_context",
    "no_interaction_structure",
    "no_relative_energy",
    "no_port_profile",
    "no_complex_phase",
)
ACTION_NAMES = ("mass", "horizontal_angle", "vertical_angle", "delay", "port")
TINY = 1e-30

# Exact column lineage from build_explicit_path_model_features.py.
FEATURE_GROUPS: dict[str, tuple[int, ...]] = {
    "delay_context": tuple(range(0, 5)),
    "angle_context": tuple(range(5, 13)),
    "relative_energy": (13, 14, 15, 19, 63),
    "interaction_structure": (
        16,
        *range(20, 43),
        *range(56, 63),
    ),
    "port_profile": tuple(range(64, 72)),
    "complex_phase": tuple(range(72, 120)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "train", "aggregate"), required=True)
    parser.add_argument("--base", choices=tuple(EXPERIMENTS), default="v11")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--outer", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_npz(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def fold_prefix(variant: str, outer: int) -> Path:
    return ARTIFACT_DIR / variant / f"outer{outer}"


def fold_report_path(variant: str, outer: int) -> Path:
    return ANALYSIS_DIR / variant / f"outer{outer}.json"


def metric_delta(
    selected: dict[str, float | int], baseline: dict[str, float | int]
) -> dict[str, float]:
    return {
        "pas": float(selected["pas"] - baseline["pas"]),
        "pdp": float(selected["pdp"] - baseline["pdp"]),
        "nmse_score": float(selected["nmse_score"] - baseline["nmse_score"]),
        "score": float(selected["score"] - baseline["score"]),
    }


class CanonicalBaseEvaluator:
    """Evaluate every endpoint against one immutable complex-channel backbone."""

    def __init__(self, device: torch.device) -> None:
        metadata = np.load(e85.OOF_META, allow_pickle=False)
        self.query_ids = metadata["query_ids"].astype(np.int64)
        self.canonical_fold = metadata["fold_id"].astype(np.int8)
        self.lookup = np.full(4000, -1, dtype=np.int64)
        self.lookup[self.query_ids] = np.arange(len(self.query_ids), dtype=np.int64)
        self.full_fold = np.load(e85.FULL_FOLD).astype(np.int8)
        self.base_name = BASE_NAME
        self.base_np = np.load(BASE_CHANNEL, mmap_mode="r")
        self.truth_np = np.load(e86.TRAIN_CHANNEL, mmap_mode="r")
        self.device = device

    def rows_for_canonical_outer(self, outer: int) -> np.ndarray:
        return self.query_ids[self.canonical_fold == outer]

    def base_metrics(self, rows: np.ndarray | None = None) -> dict[str, float | int]:
        if rows is None:
            rows = self.query_ids
        local = self.lookup[rows]
        prediction = torch.as_tensor(
            np.asarray(self.base_np[local], dtype=np.complex64), device=self.device
        )
        truth = torch.as_tensor(
            np.asarray(self.truth_np[rows], dtype=np.complex64), device=self.device
        )
        truth_pas = pas_distribution(truth)
        truth_pdp = pdp_distribution(truth)
        with torch.inference_mode():
            metrics = point_metrics(prediction, truth, truth_pas, truth_pdp)
        return statistics(*metrics)

    def channel_metrics(
        self, channel: np.ndarray, rows: np.ndarray | None = None
    ) -> dict[str, float | int]:
        if rows is None:
            rows = self.query_ids
        truth = torch.as_tensor(
            np.asarray(self.truth_np[rows], dtype=np.complex64), device=self.device
        )
        prediction = torch.as_tensor(
            np.asarray(channel, dtype=np.complex64), device=self.device
        )
        truth_pas = pas_distribution(truth)
        truth_pdp = pdp_distribution(truth)
        with torch.inference_mode():
            metrics = point_metrics(prediction, truth, truth_pas, truth_pdp)
        return statistics(*metrics)

    def evaluate_endpoint(
        self,
        rows: np.ndarray,
        endpoint_pas: np.ndarray,
        endpoint_pdp: np.ndarray,
        collect_channel: bool = False,
    ) -> tuple[dict[str, float | int], np.ndarray | None]:
        local = self.lookup[rows]
        if np.any(local < 0):
            raise ValueError("endpoint evaluation requested non-canonical rows")
        base = torch.as_tensor(
            np.array(self.base_np[local], dtype=np.complex64, copy=True),
            device=self.device,
        )
        truth = torch.as_tensor(
            np.asarray(self.truth_np[rows], dtype=np.complex64), device=self.device
        )
        truth_pas = pas_distribution(truth)
        truth_pdp = pdp_distribution(truth)
        base_pas = pas_distribution(base)
        base_pdp = pdp_distribution(base)
        spec = e86.Candidate(
            endpoint="e203",
            mode="sqrt",
            pas_beta=0.20,
            pdp_beta=0.20,
            order="pas_pdp",
            cycles=4,
        )
        with torch.inference_mode():
            candidate = e86.make_candidate(
                base,
                base_pas,
                base_pdp,
                torch.as_tensor(endpoint_pas, device=self.device),
                torch.as_tensor(endpoint_pdp, device=self.device),
                spec,
            )
            metrics = point_metrics(candidate, truth, truth_pas, truth_pdp)
        selected = statistics(*metrics)
        channel = candidate.cpu().numpy().astype(np.complex64) if collect_channel else None
        return selected, channel


class AblatedDataStore(e85.DataStore):
    def __init__(self, variant: str, device: torch.device | None = None) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.mask = np.load(e85.PATH_CACHE / "mask.npy", mmap_mode="r")
        self.resident_device = device
        self.resident: dict[str, torch.Tensor] = {}
        if device is not None:
            self._make_resident(device)

    @staticmethod
    def _resident_tensor(
        value: np.ndarray, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.as_tensor(np.array(value, copy=True), device=device, dtype=dtype)

    def _make_resident(self, device: torch.device) -> None:
        self.resident = {
            "base": self._resident_tensor(self.base, device, torch.float32),
            "fraction": self._resident_tensor(self.fraction, device, torch.float32),
            "presence": self._resident_tensor(self.presence, device, torch.float32),
            "coordinate": self._resident_tensor(self.coordinate, device, torch.float32),
            "port": self._resident_tensor(self.port, device, torch.float32),
            "mask": self._resident_tensor(self.mask, device, torch.bool),
            "target_pas": self._resident_tensor(self.target_pas, device, torch.float32),
            "target_pdp": self._resident_tensor(self.target_pdp, device, torch.float32),
        }
        if self.variant != "no_interaction_structure":
            self.resident.update(
                {
                    "sequence_geometry": self._resident_tensor(
                        self.sequence_geometry, device, torch.float32
                    ),
                    "depth": self._resident_tensor(self.depth, device, torch.int64),
                    "interactions": self._resident_tensor(
                        self.interactions, device, torch.int64
                    ),
                }
            )

    def set_coordinate(self, value: np.ndarray) -> None:
        self.coordinate = value
        if self.resident_device is not None:
            self.resident["coordinate"] = self._resident_tensor(
                value, self.resident_device, torch.float32
            )

    @property
    def removed_columns(self) -> tuple[int, ...]:
        mapping = {
            "full": (),
            "no_angle_context": FEATURE_GROUPS["angle_context"],
            "no_delay_context": FEATURE_GROUPS["delay_context"],
            "no_interaction_structure": FEATURE_GROUPS["interaction_structure"],
            "no_relative_energy": FEATURE_GROUPS["relative_energy"],
            "no_port_profile": FEATURE_GROUPS["port_profile"],
            "no_complex_phase": FEATURE_GROUPS["complex_phase"],
        }
        return mapping[self.variant]

    def batch(
        self,
        rows: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        device: torch.device,
        sequence_rows: np.ndarray | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if not self.resident:
            batch, target_pas, target_pdp = super().batch(
                rows, mean, std, device, sequence_rows
            )
        else:
            row_index = torch.as_tensor(rows, dtype=torch.int64, device=device)
            sequence_index = row_index if sequence_rows is None else torch.as_tensor(
                sequence_rows, dtype=torch.int64, device=device
            )
            fraction = self.resident["fraction"][row_index]
            presence = self.resident["presence"][row_index]
            anchored = fraction.clamp_min(1e-12) * torch.sqrt(
                presence.clamp_min(1.0) / 24.0
            )
            port = self.resident["port"][row_index].reshape(
                len(rows), e85.PATHS_PER_POINT, e85.MP * e85.N_UE
            )
            feature_mean = torch.as_tensor(mean, device=device)
            feature_std = torch.as_tensor(std, device=device)
            if self.variant == "no_interaction_structure":
                sequence_geometry = torch.empty(
                    (len(rows), e85.PATHS_PER_POINT, 1, self.sequence_geometry.shape[-1]),
                    dtype=torch.float32,
                    device=device,
                )
                depth = torch.zeros(
                    (len(rows), e85.PATHS_PER_POINT), dtype=torch.int64, device=device
                )
                interactions = torch.zeros(
                    (len(rows), e85.PATHS_PER_POINT, 1),
                    dtype=torch.int64,
                    device=device,
                )
            else:
                sequence_geometry = self.resident["sequence_geometry"][sequence_index]
                depth = self.resident["depth"][sequence_index]
                interactions = self.resident["interactions"][sequence_index]
            dummy_steps = interactions.shape[-1]
            dummy_sequence = torch.zeros(
                (len(rows), e85.PATHS_PER_POINT, dummy_steps),
                dtype=torch.int64,
                device=device,
            )
            batch = {
                "base": (self.resident["base"][row_index] - feature_mean) / feature_std,
                "base_log_mass": torch.log(anchored.clamp_min(1e-12)),
                "coordinate": self.resident["coordinate"][row_index],
                "base_log_port": torch.log(port.clamp_min(1e-8)),
                "sequence_geometry": sequence_geometry,
                "sequence_hash": torch.zeros(
                    (len(rows), e85.PATHS_PER_POINT, 2),
                    dtype=torch.int64,
                    device=device,
                ),
                "depth": depth,
                "interactions": interactions,
                "objects": dummy_sequence,
                "primitives": dummy_sequence,
            }
            target_pas = self.resident["target_pas"][row_index]
            target_pdp = self.resident["target_pdp"][row_index]
        columns = self.removed_columns
        if columns:
            batch["base"][..., list(columns)] = 0.0
        if self.variant == "no_relative_energy":
            valid = (
                self.resident["mask"][row_index]
                if self.resident
                else torch.as_tensor(np.asarray(self.mask[rows], dtype=bool), device=device)
            )
            batch["base_log_mass"] = torch.where(
                valid,
                torch.zeros((), device=device),
                torch.full((), math.log(1e-12), device=device),
            )
        if self.variant == "no_port_profile":
            batch["base_log_port"] = torch.zeros_like(batch["base_log_port"])
        return batch, target_pas, target_pdp


def load_labels(path: Path) -> e190.PathLabels:
    value = np.load(path, allow_pickle=False)
    return e190.PathLabels(
        mass=value["mass"].astype(np.float32),
        delta=value["delta"].astype(np.float32),
        port=value["port"].astype(np.float32),
        matched_count=value["matched_count"].astype(np.int16),
        matched_fraction=value["matched_fraction"].astype(np.float64),
        valid=value["valid"].astype(bool),
    )


def model_for_variant(
    store: AblatedDataStore,
    config: e190.Config,
    variant: str,
    device: torch.device,
) -> e85.SurfaceInteractionModel:
    model_config = e85.Config(
        hidden=config.hidden,
        heads=config.heads,
        sequence_layers=config.sequence_layers,
        set_layers=config.set_layers,
        feedforward=config.feedforward,
        dropout=config.dropout,
        epochs=config.epochs,
        minimum_epochs=config.minimum_epochs,
        validation_interval=config.validation_interval,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        score_limit=config.score_limit,
        port_limit=config.port_limit,
        random_state=config.random_state,
    )
    sequence_mode = "none" if variant == "no_interaction_structure" else "geometry"
    return e85.SurfaceInteractionModel(
        store.base.shape[-1],
        store.sequence_geometry.shape[-1],
        model_config,
        sequence_mode,
    ).to(device)


def train_screen(
    store: AblatedDataStore,
    labels: e190.PathLabels,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    evaluator: CanonicalBaseEvaluator,
    config: e190.Config,
    variant: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, list[dict[str, Any]]]:
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state % (2**32 - 1))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.random_state)
    model = model_for_variant(store, config, variant, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
        eta_min=config.learning_rate * 0.08,
    )
    rng = np.random.default_rng(config.random_state)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        train = e190.train_epoch(
            model, optimizer, store, labels, fit_rows, mean, std, config, device, rng
        )
        scheduler.step()
        record: dict[str, Any] = {"epoch": epoch, "train": train}
        if epoch >= config.minimum_epochs and epoch % config.validation_interval == 0:
            mass, coordinate, port = e190.endpoint_arrays(
                model, store, validation_rows, mean, std, device, config.batch_size
            )
            pas, pdp = e190.render_endpoint(mass, coordinate, port, device)
            selected, _ = evaluator.evaluate_endpoint(validation_rows, pas, pdp)
            record["validation"] = selected
            if float(selected["score"]) > best_score:
                best_score = float(selected["score"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(record)
        if epoch % config.validation_interval == 0:
            score = (
                float(record["validation"]["score"])
                if "validation" in record
                else float("nan")
            )
            print(
                f"E203 variant={variant} outer={config.outer} "
                f"epoch={epoch}/{config.epochs} validation={score:.6f}",
                flush=True,
            )
    if best_state is None or best_epoch == 0:
        raise RuntimeError(f"no validation checkpoint for {variant}/outer{config.outer}")
    return best_state, best_epoch, history


def refit(
    store: AblatedDataStore,
    labels: e190.PathLabels,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    epochs: int,
    config: e190.Config,
    variant: str,
    device: torch.device,
) -> tuple[e85.SurfaceInteractionModel, list[dict[str, float]]]:
    seed = config.random_state + 100000
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = model_for_variant(store, config, variant, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=config.learning_rate * 0.08
    )
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        values = e190.train_epoch(
            model, optimizer, store, labels, rows, mean, std, config, device, rng
        )
        scheduler.step()
        history.append({"epoch": epoch, **values})
        if epoch % config.validation_interval == 0 or epoch == epochs:
            print(
                f"E203 refit variant={variant} outer={config.outer} "
                f"epoch={epoch}/{epochs}",
                flush=True,
            )
    return model, history


def run_audit(device: torch.device) -> dict[str, Any]:
    evaluator = CanonicalBaseEvaluator(device)
    mismatch = []
    for outer in range(5):
        rows = evaluator.rows_for_canonical_outer(outer)
        mismatch.append(
            {
                "outer": outer,
                "canonical_count": int(len(rows)),
                "actually_excluded_by_e190_count": int(
                    np.sum(evaluator.full_fold[rows] == outer)
                ),
                "label_overlap_count": int(
                    np.sum(evaluator.full_fold[rows] != outer)
                ),
            }
        )
    old_channel = np.load(e190.FULL_CHANNEL, mmap_mode="r")
    old_metrics = evaluator.channel_metrics(np.asarray(old_channel))
    expected = json.loads(e190.FULL_REPORT.read_text(encoding="utf-8"))["selected"]
    identity_error = max(
        abs(float(old_metrics[key]) - float(expected[key]))
        for key in ("pas", "pdp", "nmse", "score")
    )
    report = {
        "experiment": VERSION,
        "stage": "audit",
        "status": "fold_lineage_audited",
        "base": {
            "name": evaluator.base_name,
            "path": PATHS.relative(BASE_CHANNEL),
            "metrics": evaluator.base_metrics(),
        },
        "reported_v15_replay": old_metrics,
        "reported_v15_metric_identity_error": float(identity_error),
        "fold_mismatch": mismatch,
        "total_label_overlap_count": int(sum(row["label_overlap_count"] for row in mismatch)),
        "feature_schema": {
            "dimension": 120,
            "groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
            "complex_phase_input_dimension": len(FEATURE_GROUPS["complex_phase"]),
            "complex_phase_used_as_model_input": True,
            "complex_phase_used_by_final_renderer": False,
        },
    }
    atomic_json(ANALYSIS_DIR / "lineage_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def run_fold(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if args.outer not in range(5):
        raise ValueError("--outer must be in [0, 4]")
    report_path = fold_report_path(args.variant, args.outer)
    endpoint_path = fold_prefix(args.variant, args.outer).with_name(
        fold_prefix(args.variant, args.outer).name + "_endpoint.npz"
    )
    if report_path.is_file() and endpoint_path.is_file() and not args.force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "complete":
            print(json.dumps({"status": "reused", "report": str(report_path)}, indent=2))
            return report

    started = time.time()
    timing: dict[str, float] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    config = e190.Config(
        outer=args.outer,
        validation_fold=(args.outer + 1) % 5,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    store = AblatedDataStore(args.variant, device)
    evaluator = CanonicalBaseEvaluator(device)
    held = evaluator.rows_for_canonical_outer(args.outer)
    validation = evaluator.rows_for_canonical_outer(config.validation_fold)
    all_rows = np.arange(len(store.base), dtype=np.int64)
    held_mask = np.zeros(len(store.base), dtype=bool)
    validation_mask = np.zeros(len(store.base), dtype=bool)
    held_mask[held] = True
    validation_mask[validation] = True
    screen_fit = all_rows[~held_mask & ~validation_mask]
    refit_rows = all_rows[~held_mask]
    if np.intersect1d(screen_fit, np.concatenate((held, validation))).size:
        raise RuntimeError("screen label overlap")
    if np.intersect1d(refit_rows, held).size:
        raise RuntimeError("refit label overlap")

    loaded = time.time()
    mapping = np.load(e190.MAPPING_PARAMETERS, allow_pickle=False)
    mapping_data = e98.MappingData(device)
    base_coordinate = np.array(store.coordinate, dtype=np.float32, copy=True)
    screen_coordinate = e190.corrected_coordinates(
        base_coordinate,
        mapping_data,
        mapping["screen_latent"][args.outer].astype(np.float32),
        device,
    )
    refit_coordinate = e190.corrected_coordinates(
        base_coordinate,
        mapping_data,
        mapping["refit_latent"][args.outer].astype(np.float32),
        device,
    )
    screen_labels = load_labels(
        e190.ARTIFACT_DIR / f"outer{args.outer}_screen_labels.npz"
    )
    refit_labels = load_labels(
        e190.ARTIFACT_DIR / f"outer{args.outer}_refit_labels.npz"
    )
    timing["load"] = float(time.time() - loaded)

    screen_path = fold_prefix(args.variant, args.outer).with_name(
        fold_prefix(args.variant, args.outer).name + "_screen.pt"
    )
    screen_started = time.time()
    store.set_coordinate(screen_coordinate)
    mean_screen, std_screen = store.normalization(screen_fit)
    if screen_path.is_file() and not args.force:
        checkpoint = torch.load(screen_path, map_location="cpu", weights_only=False)
        best_state = checkpoint["state_dict"]
        best_epoch = int(checkpoint["best_epoch"])
        screen_history = checkpoint["history"]
    else:
        best_state, best_epoch, screen_history = train_screen(
            store,
            screen_labels,
            screen_fit,
            validation,
            mean_screen,
            std_screen,
            evaluator,
            config,
            args.variant,
            device,
        )
        atomic_torch_save(
            screen_path,
            {
                "state_dict": best_state,
                "best_epoch": best_epoch,
                "history": screen_history,
                "feature_mean": mean_screen,
                "feature_std": std_screen,
                "variant": args.variant,
                "outer": args.outer,
            },
        )
    timing["screen"] = float(time.time() - screen_started)

    refit_started = time.time()
    store.set_coordinate(refit_coordinate)
    mean_refit, std_refit = store.normalization(refit_rows)
    model, refit_history = refit(
        store,
        refit_labels,
        refit_rows,
        mean_refit,
        std_refit,
        best_epoch,
        config,
        args.variant,
        device,
    )
    mass, coordinate, port = e190.endpoint_arrays(
        model, store, held, mean_refit, std_refit, device, config.batch_size
    )
    timing["refit_and_inference"] = float(time.time() - refit_started)

    scoring_started = time.time()
    pas, pdp = e190.render_endpoint(mass, coordinate, port, device)
    selected, _ = evaluator.evaluate_endpoint(held, pas, pdp)
    baseline = evaluator.base_metrics(held)
    timing["render_and_score"] = float(time.time() - scoring_started)
    atomic_npz(
        endpoint_path,
        query_ids=held.astype(np.int32),
        mass=mass.astype(np.float32),
        coordinate=coordinate.astype(np.float32),
        port=port.astype(np.float32),
    )
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
        peak_memory = float(torch.cuda.max_memory_allocated(device) / 2**30)
    else:
        gpu = "cpu"
        peak_memory = 0.0
    report = {
        "experiment": VERSION,
        "stage": "train",
        "status": "complete",
        "variant": args.variant,
        "outer": args.outer,
        "configuration": asdict(config),
        "split": {
            "held_count": int(len(held)),
            "validation_count": int(len(validation)),
            "screen_fit_count": int(len(screen_fit)),
            "refit_count": int(len(refit_rows)),
            "screen_label_overlap": 0,
            "refit_label_overlap": 0,
            "split_definition": "canonical OOF_META holes, not FULL_FOLD",
        },
        "best_epoch": best_epoch,
        "selected": selected,
        "base": {"name": evaluator.base_name, "metrics": baseline},
        "delta": metric_delta(selected, baseline),
        "removed_columns": list(store.removed_columns),
        "sequence_mode": model.sequence_mode,
        "training": {"screen": screen_history, "refit": refit_history},
        "output": PATHS.relative(endpoint_path),
        "performance": {
            "device": gpu,
            "peak_torch_memory_gib": peak_memory,
            "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
            "timing_seconds": timing,
            "measured_bottleneck": max(timing, key=timing.get),
        },
        "seconds": float(time.time() - started),
    }
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def assemble_endpoint(variant: str, query_ids: np.ndarray) -> dict[str, np.ndarray]:
    lookup = np.full(4000, -1, dtype=np.int64)
    lookup[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    mass = np.zeros((len(query_ids), e85.PATHS_PER_POINT), dtype=np.float32)
    coordinate = np.zeros((len(query_ids), e85.PATHS_PER_POINT, 3), dtype=np.float32)
    port = np.zeros(
        (len(query_ids), e85.PATHS_PER_POINT, e85.MP, e85.N_UE), dtype=np.float32
    )
    seen = np.zeros(len(query_ids), dtype=np.int8)
    for outer in range(5):
        path = fold_prefix(variant, outer).with_name(
            fold_prefix(variant, outer).name + "_endpoint.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        value = np.load(path, allow_pickle=False)
        rows = lookup[value["query_ids"].astype(np.int64)]
        if np.any(rows < 0) or np.any(seen[rows]):
            raise RuntimeError(f"invalid endpoint coverage for {variant}/outer{outer}")
        mass[rows] = value["mass"]
        coordinate[rows] = value["coordinate"]
        port[rows] = value["port"]
        seen[rows] += 1
    if not np.all(seen == 1):
        raise RuntimeError(f"endpoint coverage is not exactly once for {variant}")
    return {"mass": mass, "coordinate": coordinate, "port": port}


def raw_fold_endpoint(
    evaluator: CanonicalBaseEvaluator, device: torch.device
) -> dict[str, np.ndarray]:
    store = e85.DataStore()
    mapping = np.load(e190.MAPPING_PARAMETERS, allow_pickle=False)
    mapping_data = e98.MappingData(device)
    base_coordinate = np.array(store.coordinate, dtype=np.float32, copy=True)
    all_mass, all_port = e190.raw_prior(store)
    mass = np.zeros((len(evaluator.query_ids), e85.PATHS_PER_POINT), dtype=np.float32)
    coordinate = np.zeros((len(evaluator.query_ids), e85.PATHS_PER_POINT, 3), dtype=np.float32)
    port = np.zeros(
        (len(evaluator.query_ids), e85.PATHS_PER_POINT, e85.MP, e85.N_UE),
        dtype=np.float32,
    )
    for outer in range(5):
        rows = evaluator.rows_for_canonical_outer(outer)
        local = evaluator.lookup[rows]
        corrected = e190.corrected_coordinates(
            base_coordinate,
            mapping_data,
            mapping["refit_latent"][outer].astype(np.float32),
            device,
        )
        mass[local] = all_mass[rows]
        coordinate[local] = corrected[rows]
        port[local] = all_port[rows].reshape(
            len(rows), e85.PATHS_PER_POINT, e85.MP, e85.N_UE
        )
    return {"mass": mass, "coordinate": coordinate, "port": port}


def shapley(
    values: dict[int, dict[str, float | int]], metric: str, count: int
) -> dict[str, float]:
    output: dict[str, float] = {}
    factorial = math.factorial
    denominator = factorial(count)
    for index, name in enumerate(ACTION_NAMES):
        contribution = 0.0
        bit = 1 << index
        for subset in range(1 << count):
            if subset & bit:
                continue
            size = subset.bit_count()
            weight = factorial(size) * factorial(count - size - 1) / denominator
            contribution += weight * (
                float(values[subset | bit][metric]) - float(values[subset][metric])
            )
        output[name] = float(contribution)
    return output


def action_attribution(
    evaluator: CanonicalBaseEvaluator,
    raw: dict[str, np.ndarray],
    full: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    values: dict[int, dict[str, float | int]] = {}
    for subset in range(1 << len(ACTION_NAMES)):
        mass = full["mass"] if subset & 1 else raw["mass"]
        coordinate = np.array(raw["coordinate"], copy=True)
        for axis, bit in enumerate((2, 4, 8)):
            if subset & bit:
                coordinate[..., axis] = full["coordinate"][..., axis]
        port = full["port"] if subset & 16 else raw["port"]
        pas, pdp = e190.render_endpoint(mass, coordinate, port, device)
        selected, _ = evaluator.evaluate_endpoint(evaluator.query_ids, pas, pdp)
        values[subset] = selected
        print(
            f"E203 action subset={subset:02d}/31 score={float(selected['score']):.6f}",
            flush=True,
        )
    contributions = {
        metric: shapley(values, metric, len(ACTION_NAMES))
        for metric in ("pas", "pdp", "nmse_score", "score")
    }
    closure = {}
    for metric in contributions:
        explained = sum(contributions[metric].values())
        actual = float(values[31][metric]) - float(values[0][metric])
        closure[metric] = {
            "actual": actual,
            "shapley_sum": explained,
            "absolute_error": abs(actual - explained),
        }
    return {
        "reference": "raw same-position physical endpoint",
        "raw": values[0],
        "full": values[31],
        "combinations": {str(key): value for key, value in values.items()},
        "shapley": contributions,
        "closure": closure,
    }


def aggregate(device: torch.device) -> dict[str, Any]:
    started = time.time()
    evaluator = CanonicalBaseEvaluator(device)
    base = evaluator.base_metrics()
    variant_rows: dict[str, Any] = {}
    endpoints: dict[str, dict[str, np.ndarray]] = {}
    for variant in VARIANTS:
        endpoint = assemble_endpoint(variant, evaluator.query_ids)
        endpoints[variant] = endpoint
        pas, pdp = e190.render_endpoint(
            endpoint["mass"], endpoint["coordinate"], endpoint["port"], device
        )
        selected, _ = evaluator.evaluate_endpoint(evaluator.query_ids, pas, pdp)
        folds = []
        for outer in range(5):
            fold_report = json.loads(
                fold_report_path(variant, outer).read_text(encoding="utf-8")
            )
            fold_base = fold_report.get("base")
            if isinstance(fold_base, dict) and "metrics" in fold_base:
                fold_base_score = float(fold_base["metrics"]["score"])
            else:
                fold_base_score = float(fold_report["v11"]["score"])
            folds.append(
                {
                    "outer": outer,
                    "count": fold_report["split"]["held_count"],
                    "score": fold_report["selected"]["score"],
                    "base_score": fold_base_score,
                    "delta": fold_report["delta"]["score"],
                    "best_epoch": fold_report["best_epoch"],
                }
            )
        variant_rows[variant] = {
            "metrics": selected,
            "delta_vs_base": metric_delta(selected, base),
            "folds": folds,
        }

    full_metrics = variant_rows["full"]["metrics"]
    for variant in VARIANTS[1:]:
        variant_rows[variant]["conditional_contribution_of_removed_information"] = (
            metric_delta(full_metrics, variant_rows[variant]["metrics"])
        )

    raw = raw_fold_endpoint(evaluator, device)
    actions = action_attribution(evaluator, raw, endpoints["full"], device)
    old_channel = np.load(e190.FULL_CHANNEL, mmap_mode="r")
    old_v15 = evaluator.channel_metrics(np.asarray(old_channel))
    report = {
        "experiment": VERSION,
        "stage": "aggregate",
        "status": "complete",
        "measurement": {
            "base": {
                "name": evaluator.base_name,
                "path": PATHS.relative(BASE_CHANNEL),
                "metrics": base,
            },
            "historical_reported_v15_with_fold_mismatch": old_v15,
            "corrected_strict_full": full_metrics,
            "historical_v15_minus_strict_full": metric_delta(old_v15, full_metrics),
            "comparison_note": (
                "Same-base fold-leakage comparison"
                if evaluator.base_name == "v11"
                else "Reference only: historical V15 used V11, strict full uses V7"
            ),
        },
        "input_information_ablation": variant_rows,
        "output_action_attribution": actions,
        "feature_groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
        "seconds": float(time.time() - started),
        "performance": {
            "device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
            "peak_torch_memory_gib": float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0,
            "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
        },
    }
    maximum_closure_error = max(
        value["absolute_error"] for value in actions["closure"].values()
    )
    report["checks"] = {
        "query_count": int(len(evaluator.query_ids)),
        "all_variants_complete": True,
        "maximum_shapley_closure_error": float(maximum_closure_error),
        "shapley_closure_passed": bool(maximum_closure_error <= 1e-6),
    }
    atomic_json(FINAL_REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    args = parse_args()
    configure_experiment(args.base)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    if args.stage == "audit":
        run_audit(device)
    elif args.stage == "train":
        run_fold(args, device)
    else:
        aggregate(device)


if __name__ == "__main__":
    main()
