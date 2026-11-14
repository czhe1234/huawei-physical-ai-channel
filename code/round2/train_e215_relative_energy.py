#!/usr/bin/env python3
"""Train the E214 base/seed-42 model with one relative-energy feature."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
import torch
from torch import nn

from project_paths import PATHS
import build_e214_geometry_path_topology as topology
import build_e215_relative_energy as energy_builder
import train_e214_geometry_path_topology as e214


VERSION = "Round2-E215-relative-energy-diagnostic"
VARIANT = "base_s1_relative_energy"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e215_relative_energy"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e215_relative_energy"
PREPARE_REPORT = ANALYSIS_DIR / "prepare_report.json"
ENERGY_PATH = energy_builder.ENERGY_PATH
E214_TOPOLOGY_AUDIT = topology.ANALYSIS_DIR / "topology_audit.json"
E214_ALLOWED_FIELDS = tuple(topology.ALLOWED_FIELDS)
E215_ALLOWED_FIELDS = E214_ALLOWED_FIELDS + ("relative_energy",)
UNUSED_PROPAGATION_FIELDS = energy_builder.UNUSED_PROPAGATION_FIELDS

BaseFeatureStore = e214.FeatureStore
BaseGeometryPathModel = e214.GeometryPathModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "aggregate", "status"), required=True)
    parser.add_argument("--outer", type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=e214.Config.epochs)
    parser.add_argument("--batch-size", type=int, default=e214.Config.batch_size)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def log_energy(value: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(value.astype(np.float32), np.float32(1e-20)))


class RelativeEnergyFeatureStore(BaseFeatureStore):
    def __init__(self, variant: str, device: torch.device) -> None:
        if variant != VARIANT:
            raise ValueError(f"E215 only supports {VARIANT}, got {variant}")
        super().__init__(variant, device)
        self.relative_energy = np.load(ENERGY_PATH, mmap_mode="r")
        mask = self.topology.full("mask").astype(bool)
        if self.relative_energy.shape != mask.shape:
            raise RuntimeError(
                f"relative-energy shape mismatch: {self.relative_energy.shape} != {mask.shape}"
            )
        valid = np.asarray(self.relative_energy)[mask]
        if not np.all(np.isfinite(valid)) or np.any(valid < 0.0):
            raise FloatingPointError("relative-energy feature is invalid")
        if np.any(np.asarray(self.relative_energy)[~mask] != 0.0):
            raise RuntimeError("relative-energy padding is nonzero")

    def normalization(self, rows: np.ndarray) -> dict[str, Any]:
        output: dict[str, Any] = super().normalization(rows)
        mask = self.topology.get("mask", rows).astype(bool)
        values = log_energy(np.asarray(self.relative_energy[rows]))[mask]
        if len(values) == 0:
            raise RuntimeError("no relative-energy values in normalization rows")
        mean = float(np.mean(values, dtype=np.float64))
        std = float(np.std(values, dtype=np.float64))
        if std < 1e-4:
            std = 1.0
        output["relative_energy_log10_mean"] = mean
        output["relative_energy_log10_std"] = std
        return output

    def batch(
        self,
        rows: np.ndarray,
        normalization: dict[str, Any],
        base_pas: torch.Tensor | np.ndarray | None = None,
        base_pdp: torch.Tensor | np.ndarray | None = None,
        include_targets: bool = True,
    ) -> dict[str, torch.Tensor]:
        output = super().batch(
            rows,
            normalization,
            base_pas=base_pas,
            base_pdp=base_pdp,
            include_targets=include_targets,
        )
        rows = np.asarray(rows, dtype=np.int64)
        raw_mask = self.topology.get("mask", rows).astype(bool)
        raw = np.asarray(self.relative_energy[rows], dtype=np.float32)
        transformed = (
            log_energy(raw) - float(normalization["relative_energy_log10_mean"])
        ) / float(normalization["relative_energy_log10_std"])
        transformed[~raw_mask] = 0.0
        feature = torch.as_tensor(transformed[..., None], device=self.device)
        output["continuous"] = torch.cat((output["continuous"], feature), dim=2)
        output["relative_energy"] = torch.as_tensor(raw, device=self.device)
        return output


class RelativeEnergyModel(BaseGeometryPathModel):
    def __init__(self, point_dim: int, config: e214.Config) -> None:
        super().__init__(point_dim, config)
        old = self.path_encoder[0]
        if not isinstance(old, nn.Linear):
            raise TypeError("unexpected E214 path encoder")
        rng_state = torch.random.get_rng_state()
        expanded = nn.Linear(old.in_features + 1, old.out_features, bias=old.bias is not None)
        torch.random.set_rng_state(rng_state)
        with torch.no_grad():
            expanded.weight[:, : old.in_features].copy_(old.weight)
            expanded.weight[:, old.in_features].zero_()
            if old.bias is not None and expanded.bias is not None:
                expanded.bias.copy_(old.bias)
        self.path_encoder[0] = expanded


def e215_topology_audit() -> dict[str, object]:
    if not E214_TOPOLOGY_AUDIT.is_file():
        raise FileNotFoundError(E214_TOPOLOGY_AUDIT)
    base_audit = json.loads(E214_TOPOLOGY_AUDIT.read_text(encoding="utf-8"))
    if base_audit.get("status") != "complete" or base_audit.get("forbidden_fields_accessed"):
        raise RuntimeError("E214 frozen topology audit failed")
    if not energy_builder.STATE_PATH.is_file():
        raise FileNotFoundError(energy_builder.STATE_PATH)
    energy_audit = json.loads(energy_builder.STATE_PATH.read_text(encoding="utf-8"))
    if energy_audit.get("status") != "complete":
        raise RuntimeError("E215 relative-energy alignment is incomplete")
    lineage = energy_audit.get("data_lineage", {})
    if lineage.get("added_field") != "relative_energy":
        raise RuntimeError("E215 added-field lineage changed")
    if lineage.get("complex_fields_accessed"):
        raise RuntimeError("E215 unexpectedly accessed complex coefficients")
    return {
        "status": "complete",
        "allowed_fields": list(E215_ALLOWED_FIELDS),
        "forbidden_fields_accessed": [],
        "unused_propagation_fields": list(UNUSED_PROPAGATION_FIELDS),
        "frozen_topology_audit": PATHS.relative(E214_TOPOLOGY_AUDIT),
        "relative_energy_audit": PATHS.relative(energy_builder.STATE_PATH),
    }


def finite_gradient_check(config: e214.Config, device: torch.device) -> dict[str, object]:
    e214.set_seed(config.random_state)
    store = RelativeEnergyFeatureStore(VARIANT, device)
    rows = np.arange(256, dtype=np.int64)
    normalization = store.normalization(rows)
    active = store.active_rows(rows)[: min(4, config.batch_size)]
    if len(active) == 0:
        raise RuntimeError("E215 finite-gradient check found no active rows")
    model = RelativeEnergyModel(store.point_raw.shape[1], config).to(device)
    batch = store.batch(active, normalization)
    output = model(batch)
    loss, _ = e214.training_objective(output, batch, config)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite = bool(torch.isfinite(loss)) and bool(gradients) and all(
        bool(torch.all(torch.isfinite(item))) for item in gradients
    )
    result = {
        "finite": finite,
        "loss": float(loss.detach().item()),
        "batch": int(len(active)),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "added_parameters": int(config.hidden),
        "injection": "one standardized log10(relative_energy) path feature",
        "direct_mass_prior": False,
    }
    del model, store, batch, output, loss, gradients
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not finite:
        raise FloatingPointError("E215 finite-gradient check failed")
    return result


def relative_energy_diagnostic() -> dict[str, object]:
    energy = np.load(ENERGY_PATH, mmap_mode="r")
    topology_arrays = e214.TopologyArrays(VARIANT)
    mask = topology_arrays.full("mask").astype(bool)
    depth = topology_arrays.full("depth").astype(np.float64)
    coordinate = topology_arrays.full("coordinate").astype(np.float64)
    with np.load(e214.labels_path(VARIANT), allow_pickle=False) as payload:
        target_mass = payload["target_mass"].astype(np.float64)
        assigned = payload["assigned"].astype(bool)
    energy_log = log_energy(np.asarray(energy))[mask].astype(np.float64)
    target_log = np.log10(np.maximum(target_mass, 1e-20))[mask]
    assigned_flat = assigned[mask]
    ranks = rankdata(energy_log)
    positive = ranks[assigned_flat]
    positive_count = int(np.sum(assigned_flat))
    negative_count = int(np.sum(~assigned_flat))
    auc = float(
        (positive.sum() - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )
    rows = np.flatnonzero(target_mass.sum(axis=1) > 0.0)
    energy_cross_entropy: list[float] = []
    old_cross_entropy: list[float] = []
    for row in rows:
        ids = np.flatnonzero(mask[row])
        target = target_mass[row, ids]
        target /= target.sum()
        energy_prior = np.maximum(np.asarray(energy[row, ids], dtype=np.float64), 1e-12)
        energy_prior /= energy_prior.sum()
        old_prior = np.exp(-0.12 * depth[row, ids] - 0.002 * coordinate[row, ids, 2])
        old_prior /= old_prior.sum()
        energy_cross_entropy.append(float(-np.sum(target * np.log(energy_prior))))
        old_cross_entropy.append(float(-np.sum(target * np.log(old_prior))))
    return {
        "path_count": int(np.sum(mask)),
        "assigned_count": positive_count,
        "assigned_fraction": float(np.mean(assigned_flat)),
        "log_energy_vs_log_target_mass_pearson": float(
            pearsonr(energy_log, target_log).statistic
        ),
        "log_energy_vs_log_target_mass_spearman": float(
            spearmanr(energy_log, target_log).statistic
        ),
        "assigned_auc": auc,
        "mean_energy_assigned": float(np.mean(np.asarray(energy)[assigned])),
        "mean_energy_unassigned": float(np.mean(np.asarray(energy)[mask & ~assigned])),
        "direct_prior_cross_entropy": float(np.mean(energy_cross_entropy)),
        "e214_prior_cross_entropy": float(np.mean(old_cross_entropy)),
        "injection_decision": "feature_only_due_to_weak_alignment",
    }


def configure_e214() -> None:
    e214.VERSION = VERSION
    e214.ARTIFACT_DIR = ARTIFACT_DIR
    e214.ANALYSIS_DIR = ANALYSIS_DIR
    e214.PREPARE_REPORT = PREPARE_REPORT
    e214.VARIANTS = (VARIANT,)
    e214.VARIANT_INFO[VARIANT] = {"seed_mode": "s1", "scatter": False}
    e214.FeatureStore = RelativeEnergyFeatureStore
    e214.GeometryPathModel = RelativeEnergyModel
    e214.topology_audit = e215_topology_audit
    e214.finite_gradient_check = finite_gradient_check
    topology.ALLOWED_FIELDS = E215_ALLOWED_FIELDS
    topology.FORBIDDEN_FIELDS = UNUSED_PROPAGATION_FIELDS


def enrich_lineage(payload: dict[str, Any]) -> dict[str, Any]:
    lineage = payload.setdefault("data_lineage", {})
    lineage.update(
        {
            "added_rt_field": "relative_energy",
            "relative_energy_seed": energy_builder.SEED,
            "relative_energy_source": energy_builder.SOURCE,
            "complex_rt_fields_used": [],
            "unused_propagation_fields": list(UNUSED_PROPAGATION_FIELDS),
            "injection": "standardized_log10_feature_only",
            "direct_mass_prior": False,
        }
    )
    return payload


def prepare(config: e214.Config, device: torch.device, force: bool) -> dict[str, Any]:
    if not ENERGY_PATH.is_file():
        energy_builder.build(force=False)
    result = e214.prepare(config, device, force)
    result["relative_energy_diagnostic"] = relative_energy_diagnostic()
    result["data_lineage"] = {
        "frozen_candidate_set": "E214/base_s1",
        "added_rt_field": "relative_energy",
        "allowed_rt_fields": list(E215_ALLOWED_FIELDS),
        "unused_propagation_fields": list(UNUSED_PROPAGATION_FIELDS),
        "complex_rt_fields_used": [],
        "injection": "standardized_log10_feature_only",
        "direct_mass_prior": False,
    }
    result["topology_audit"] = e215_topology_audit()
    atomic_json(PREPARE_REPORT, result)
    return result


def train_outer(
    outer: int, config: e214.Config, device: torch.device, force: bool
) -> dict[str, Any]:
    result = e214.train_outer(VARIANT, outer, config, device, force)
    report_path = e214.fold_report_path(VARIANT, outer)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    enrich_lineage(report)
    atomic_json(report_path, report)
    return report


def aggregate(config: e214.Config, device: torch.device) -> dict[str, Any]:
    report = e214.aggregate_variant(VARIANT, config, device)
    enrich_lineage(report)
    report["comparison_contract"] = {
        "control": "E214/base_s1",
        "changed_input": "relative_energy only",
        "candidate_set_changed": False,
        "training_budget_changed": False,
        "direct_energy_prior": False,
    }
    atomic_json(e214.final_report_path(VARIANT), report)
    return report


def status(config: e214.Config) -> dict[str, object]:
    return {
        "experiment": VERSION,
        "prepare": PREPARE_REPORT.is_file(),
        "alignment": ENERGY_PATH.is_file() and energy_builder.STATE_PATH.is_file(),
        "folds": [e214.fold_report_path(VARIANT, outer).is_file() for outer in range(5)],
        "aggregate": e214.final_report_path(VARIANT).is_file(),
        "config": e214.config_payload(config),
    }


def main() -> None:
    args = parse_args()
    config = e214.Config(epochs=args.epochs, batch_size=args.batch_size)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    configure_e214()
    try:
        if args.stage == "prepare":
            result: object = prepare(config, device, args.force)
        elif args.stage == "train":
            if args.outer is None:
                raise ValueError("train requires --outer")
            result = train_outer(args.outer, config, device, args.force)
        elif args.stage == "aggregate":
            result = aggregate(config, device)
        else:
            result = status(config)
        print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False), flush=True)
    except Exception as error:
        suffix = "none" if args.outer is None else str(args.outer)
        failure_path = ANALYSIS_DIR / f"failure_{args.stage}_outer{suffix}.json"
        payload = {
            "experiment": VERSION,
            "status": "failed",
            "stage": args.stage,
            "outer": args.outer,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "exit_code": 1,
        }
        atomic_json(failure_path, payload)
        raise


if __name__ == "__main__":
    main()
