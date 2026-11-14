#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import nn

from data import load_positions
from project_paths import PATHS
from round2.spatial_hole_cv import build_spatial_hole_folds
import build_e217_angle_delay_support as support
import train_e213_no_rt_spectral_residual as e213
import train_e214_geometry_path_topology as e214


VERSION = "Round2-E217-angle-delay-support-only"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e217_angle_delay_support_only"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e217_angle_delay_support_only"
PREPARE_REPORT = ANALYSIS_DIR / "prepare_report.json"
CANONICAL_META = ARTIFACT_DIR / "canonical_fixed500_meta.npz"
BROAD_FOLD = ARTIFACT_DIR / "broad_fold_id_i8.npy"
TRAIN_BASE_PAS = ARTIFACT_DIR / "train_v7_compact_pas_f32.npy"
TRAIN_BASE_PDP = ARTIFACT_DIR / "train_v7_compact_pdp_f32.npy"
CANONICAL_BASE_PAS = ARTIFACT_DIR / "canonical_v7_compact_pas_f32.npy"
CANONICAL_BASE_PDP = ARTIFACT_DIR / "canonical_v7_compact_pdp_f32.npy"
OFFICIAL_PAS = ARTIFACT_DIR / "official_train_compact_pas_f32.npy"
OFFICIAL_PDP = ARTIFACT_DIR / "official_train_compact_pdp_f32.npy"
OFFICIAL_POWER = ARTIFACT_DIR / "official_train_power_f64.npy"
TRAIN_BASE_H = (
    PATHS.artifacts
    / "round2"
    / "e160_full4000_v7_supervision"
    / "full4000_v7_core.npy"
)
CANONICAL_BASE_H = (
    PATHS.artifacts
    / "round2"
    / "e160_full4000_v7_supervision"
    / "canonical_v7_complete.npy"
)
REFERENCE_ENDPOINT_DIR = (
    PATHS.root
    / "final_code"
    / "experiments"
    / "v16_rt_ablation"
    / "artifacts"
    / "strict_pathstate"
    / "full"
)
VARIANT = "support_only"
FOURIER_FREQUENCIES = (1.0, 2.0, 4.0, 8.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("folds", "prepare", "train", "aggregate", "status"), required=True)
    parser.add_argument("--outer", type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=32)
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


def atomic_npz(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def build_fold_assets(force: bool = False) -> dict[str, object]:
    if CANONICAL_META.is_file() and BROAD_FOLD.is_file() and not force:
        with np.load(CANONICAL_META, allow_pickle=False) as payload:
            return {
                "status": "reused",
                "query_count": int(len(payload["query_ids"])),
                "canonical_counts": np.bincount(payload["fold_id"], minlength=5).astype(int).tolist(),
                "broad_counts": np.bincount(np.load(BROAD_FOLD), minlength=5).astype(int).tolist(),
            }
    setup, train_positions, test_positions, train_site, test_site = load_positions()
    del setup
    folds = build_spatial_hole_folds(
        train_positions, test_positions, train_site, test_site
    )
    query_parts: list[np.ndarray] = []
    fold_parts: list[np.ndarray] = []
    identity = []
    for fold in folds:
        reference_path = REFERENCE_ENDPOINT_DIR / f"outer{fold.fold}_endpoint.npz"
        with np.load(reference_path, allow_pickle=False) as reference:
            expected = reference["query_ids"].astype(np.int64)
        exact = np.array_equal(fold.query_indices, expected)
        identity.append(exact)
        if not exact:
            raise RuntimeError(f"canonical fixed500 identity mismatch for outer {fold.fold}")
        query_parts.append(fold.query_indices.astype(np.int32))
        fold_parts.append(np.full(len(fold.query_indices), fold.fold, dtype=np.int8))
    query_ids = np.concatenate(query_parts)
    fold_id = np.concatenate(fold_parts)
    order = np.argsort(query_ids, kind="stable")
    query_ids = query_ids[order]
    fold_id = fold_id[order]
    broad = np.full(len(train_positions), -1, dtype=np.int8)
    for site in sorted(np.unique(train_site).tolist()):
        ids = np.flatnonzero(train_site == site)
        xy = np.asarray(train_positions[ids, :2], dtype=np.float64)
        model = KMeans(
            n_clusters=160,
            random_state=20_260_806_66 + int(site),
            n_init=16,
        ).fit(xy)
        centers = model.cluster_centers_
        cluster_order = np.lexsort((centers[:, 1], centers[:, 0]))
        cluster_fold = np.empty(160, dtype=np.int8)
        cluster_fold[cluster_order] = np.arange(160, dtype=np.int64) % 5
        broad[ids] = cluster_fold[model.labels_]
    broad_counts = np.bincount(broad, minlength=5).astype(int).tolist()
    if broad_counts != [771, 806, 791, 801, 831]:
        raise RuntimeError(f"historical broad-fold identity mismatch: {broad_counts}")
    atomic_npz(CANONICAL_META, query_ids=query_ids, fold_id=fold_id)
    np.save(BROAD_FOLD, broad)
    report = {
        "experiment": VERSION,
        "status": "complete",
        "canonical_query_count": int(len(query_ids)),
        "canonical_counts": np.bincount(fold_id, minlength=5).astype(int).tolist(),
        "reference_endpoint_identity": identity,
        "broad_counts": broad_counts,
        "canonical_query_ids": PATHS.relative(CANONICAL_META),
        "broad_fold": PATHS.relative(BROAD_FOLD),
        "exit_code": 0,
    }
    atomic_json(ANALYSIS_DIR / "fold_identity.json", report)
    return report


def configure() -> None:
    e213.TRAIN_BASE_H = TRAIN_BASE_H
    e213.CANONICAL_BASE_H = CANONICAL_BASE_H
    e213.OOF_META = CANONICAL_META
    e213.FULL_FOLD = BROAD_FOLD
    e213.TRAIN_BASE_PAS = TRAIN_BASE_PAS
    e213.TRAIN_BASE_PDP = TRAIN_BASE_PDP
    e213.CANONICAL_BASE_PAS = CANONICAL_BASE_PAS
    e213.CANONICAL_BASE_PDP = CANONICAL_BASE_PDP
    e213.OFFICIAL_PAS = OFFICIAL_PAS
    e213.OFFICIAL_PDP = OFFICIAL_PDP
    e213.OFFICIAL_POWER = OFFICIAL_POWER
    e214.VERSION = VERSION
    e214.ARTIFACT_DIR = ARTIFACT_DIR
    e214.ANALYSIS_DIR = ANALYSIS_DIR
    e214.PREPARE_REPORT = PREPARE_REPORT
    e214.TRUTH_COMPONENTS = ARTIFACT_DIR / "official_h_top128_components.npz"
    e214.VARIANTS = (VARIANT,)
    e214.VARIANT_INFO = {VARIANT: {"seed_mode": "s1", "scatter": False}}
    e214.topology.ALLOWED_FIELDS = ("candidate_coordinate", "candidate_mask")
    e214.FeatureStore = CoordinateFeatureStore
    e214.GeometryPathModel = CoordinateOnlyModel
    e214.TopologyArrays = CoordinateArrays
    e214.topology_audit = topology_audit
    e214.finite_gradient_check = finite_gradient_check


def compact_cache(
    source_path: Path,
    pas_path: Path,
    pdp_path: Path,
    power_path: Path | None,
    base: bool,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, object]:
    source = np.load(source_path, mmap_mode="r")
    expected_pas = (len(source), e214.MP, e214.MH, e214.MV, e214.N_UE)
    expected_pdp = (len(source), e214.MP, e214.N_UE, e214.S_SC)
    ready = pas_path.is_file() and pdp_path.is_file() and (
        power_path is None or power_path.is_file()
    )
    if ready and not force:
        if np.load(pas_path, mmap_mode="r").shape == expected_pas and np.load(
            pdp_path, mmap_mode="r"
        ).shape == expected_pdp:
            return {"status": "reused", "count": int(len(source))}
    pas_path.parent.mkdir(parents=True, exist_ok=True)
    partial_pas = pas_path.with_name(f".{pas_path.name}.partial-{os.getpid()}")
    partial_pdp = pdp_path.with_name(f".{pdp_path.name}.partial-{os.getpid()}")
    partial_power = (
        None
        if power_path is None
        else power_path.with_name(f".{power_path.name}.partial-{os.getpid()}")
    )
    pas_out = np.lib.format.open_memmap(
        partial_pas, mode="w+", dtype=np.float32, shape=expected_pas
    )
    pdp_out = np.lib.format.open_memmap(
        partial_pdp, mode="w+", dtype=np.float32, shape=expected_pdp
    )
    power_out = (
        None
        if partial_power is None
        else np.lib.format.open_memmap(
            partial_power, mode="w+", dtype=np.float64, shape=(len(source),)
        )
    )
    started = time.time()
    for start in range(0, len(source), batch_size):
        end = min(start + batch_size, len(source))
        channel = torch.as_tensor(
            np.array(source[start:end], dtype=np.complex64, copy=True), device=device
        )
        if base:
            pas, pdp, power = e213.base_compact_spectra(channel)
        else:
            pas, pdp, power = e213.compact_spectra(channel)
        pas_out[start:end] = pas.cpu().numpy()
        pdp_out[start:end] = pdp.cpu().numpy()
        if power_out is not None:
            power_out[start:end] = power.cpu().numpy()
        if end == len(source) or end % 500 == 0:
            pas_out.flush()
            pdp_out.flush()
            if power_out is not None:
                power_out.flush()
            print(f"E217 compact {'base' if base else 'official'} {end}/{len(source)}", flush=True)
    pas_out.flush()
    pdp_out.flush()
    if power_out is not None:
        power_out.flush()
    del pas_out, pdp_out, power_out
    os.replace(partial_pas, pas_path)
    os.replace(partial_pdp, pdp_path)
    if partial_power is not None and power_path is not None:
        os.replace(partial_power, power_path)
    return {
        "status": "created",
        "count": int(len(source)),
        "seconds": float(time.time() - started),
    }


class CoordinateArrays:
    def __init__(self, variant: str) -> None:
        if variant != VARIANT:
            raise ValueError(variant)
        self.coordinate = np.load(support.COORDINATE_PATH, mmap_mode="r")
        self.mask = np.load(support.MASK_PATH, mmap_mode="r")

    def get(self, name: str, rows: np.ndarray) -> np.ndarray:
        if name == "coordinate":
            return np.asarray(self.coordinate[rows])
        if name == "mask":
            return np.asarray(self.mask[rows])
        raise KeyError(name)

    def full(self, name: str) -> np.ndarray:
        return self.get(name, np.arange(4000, dtype=np.int64))


def topology_audit() -> dict[str, object]:
    report = json.loads(support.SUPPORT_REPORT.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("selection_uses_energy") is not False:
        raise RuntimeError("E217 support audit failed")
    if report.get("model_visible_fields") != ["coordinate", "mask"]:
        raise RuntimeError("E217 model input boundary changed")
    return {
        "status": "complete",
        "allowed_fields": ["candidate_coordinate", "candidate_mask"],
        "forbidden_fields_accessed": [],
    }


class CoordinateFeatureStore:
    def __init__(self, variant: str, device: torch.device) -> None:
        self.variant = variant
        self.device = device
        self.topology = CoordinateArrays(variant)
        with np.load(e214.labels_path(variant), allow_pickle=False) as payload:
            self.labels = {name: np.array(payload[name], copy=True) for name in payload.files}
        topology_mask = self.topology.full("mask").astype(bool)
        if not np.array_equal(topology_mask, self.labels["candidate_mask"]):
            raise RuntimeError("candidate identity changed")
        self.mask_count = topology_mask.sum(axis=1).astype(np.int16)
        self.point_raw = np.empty((4000, 0), dtype=np.float32)
        self.base_pas = torch.as_tensor(np.asarray(np.load(TRAIN_BASE_PAS, mmap_mode="r")), device=device)
        self.base_pdp = torch.as_tensor(np.asarray(np.load(TRAIN_BASE_PDP, mmap_mode="r")), device=device)
        self.target_pas = torch.as_tensor(np.asarray(np.load(OFFICIAL_PAS, mmap_mode="r")), device=device)
        self.target_pdp = torch.as_tensor(np.asarray(np.load(OFFICIAL_PDP, mmap_mode="r")), device=device)
        self.power = np.asarray(np.load(OFFICIAL_POWER, mmap_mode="r"), dtype=np.float64)

    def normalization(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        del rows
        return {}

    def active_rows(self, rows: np.ndarray) -> np.ndarray:
        return rows[(self.power[rows] > 0.0) & (self.mask_count[rows] > 0)]

    def batch(
        self,
        rows: np.ndarray,
        normalization: dict[str, np.ndarray],
        base_pas: torch.Tensor | np.ndarray | None = None,
        base_pdp: torch.Tensor | np.ndarray | None = None,
        include_targets: bool = True,
    ) -> dict[str, torch.Tensor]:
        del normalization
        rows = np.asarray(rows, dtype=np.int64)
        mask = self.topology.get("mask", rows).astype(bool)
        has_paths = mask.any(axis=1)
        if np.any(~has_paths):
            mask[~has_paths, 0] = True
        coordinate = self.topology.get("coordinate", rows).astype(np.float32)
        row_tensor = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        output: dict[str, torch.Tensor] = {
            "rows": row_tensor,
            "mask": torch.as_tensor(mask, device=self.device),
            "has_paths": torch.as_tensor(has_paths, device=self.device),
            "coordinate": torch.as_tensor(coordinate, device=self.device),
            "base_pas": self.base_pas[row_tensor] if base_pas is None else torch.as_tensor(base_pas, device=self.device),
            "base_pdp": self.base_pdp[row_tensor] if base_pdp is None else torch.as_tensor(base_pdp, device=self.device),
        }
        if include_targets:
            output.update(
                {
                    "target_pas": self.target_pas[row_tensor],
                    "target_pdp": self.target_pdp[row_tensor],
                    "target_mass": torch.as_tensor(self.labels["target_mass"][rows], device=self.device),
                    "target_delta": torch.zeros((len(rows), e214.PATHS_PER_POINT, 3), device=self.device),
                    "target_port": torch.as_tensor(self.labels["target_port"][rows], device=self.device),
                    "assigned": torch.as_tensor(self.labels["assigned"][rows], device=self.device),
                    "label_valid": torch.as_tensor(self.labels["valid"][rows], device=self.device),
                }
            )
        return output


class CoordinateOnlyModel(nn.Module):
    def __init__(self, point_dim: int, config: e214.Config) -> None:
        super().__init__()
        del point_dim
        self.config = config
        input_dim = 3 * 2 * len(FOURIER_FREQUENCIES)
        self.input = nn.Sequential(
            nn.Linear(input_dim, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            layer, num_layers=config.set_layers, enable_nested_tensor=False
        )
        self.mass_head = nn.Linear(config.hidden, 1)
        self.port_head = nn.Linear(config.hidden, e214.PORTS)
        self.confidence_head = nn.Linear(config.hidden, 2)
        nn.init.zeros_(self.mass_head.weight)
        nn.init.zeros_(self.mass_head.bias)
        nn.init.zeros_(self.port_head.weight)
        nn.init.zeros_(self.port_head.bias)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.constant_(self.confidence_head.bias, -1.5)

    def coordinate_features(self, coordinate: torch.Tensor) -> torch.Tensor:
        normalized = coordinate / torch.as_tensor(
            e214.PERIODS, device=coordinate.device, dtype=coordinate.dtype
        )
        values = []
        for frequency in FOURIER_FREQUENCIES:
            phase = 2.0 * torch.pi * frequency * normalized
            values.extend((torch.sin(phase), torch.cos(phase)))
        return torch.cat(values, dim=2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden = self.input(self.coordinate_features(batch["coordinate"]))
        hidden = self.set_encoder(hidden, src_key_padding_mask=~batch["mask"])
        mass_logits = self.mass_head(hidden)[..., 0].masked_fill(~batch["mask"], -1e4)
        mass = torch.softmax(mass_logits, dim=1)
        port_logits = self.port_head(hidden)
        port = torch.softmax(port_logits, dim=2).reshape(
            len(hidden), e214.PATHS_PER_POINT, e214.MP, e214.N_UE
        )
        rendered_pas, rendered_pdp = e214.render_spectra(
            mass, batch["coordinate"], port
        )
        pooled = e214.masked_mean(hidden, batch["mask"], dim=1)
        confidence = torch.sigmoid(self.confidence_head(pooled))
        base_pas = e213.normalize_compact_pas(batch["base_pas"].float())
        base_pdp = e213.normalize_compact_pdp(batch["base_pdp"].float())
        pas = e214.blend_pas(base_pas, rendered_pas, confidence[:, 0])
        pdp = e214.blend_pdp(base_pdp, rendered_pdp, confidence[:, 1])
        usable = batch["has_paths"] & (base_pas.sum(dim=(1, 2, 3, 4)) > 0.0)
        pas = torch.where(usable[:, None, None, None, None], pas, batch["base_pas"].float())
        pdp = torch.where(usable[:, None, None, None], pdp, batch["base_pdp"].float())
        coordinate_residual = torch.zeros_like(batch["coordinate"])
        return {
            "pas": pas.float(),
            "pdp": pdp.float(),
            "mass": mass.float(),
            "mass_logits": mass_logits.float(),
            "mass_residual": mass_logits.masked_fill(~batch["mask"], 0.0).float(),
            "coordinate": batch["coordinate"].float(),
            "coordinate_residual": coordinate_residual,
            "port": port.float(),
            "port_logits": port_logits.float(),
            "gate": confidence.float(),
        }


def finite_gradient_check(config: e214.Config, device: torch.device) -> dict[str, object]:
    e214.set_seed(config.random_state)
    store = CoordinateFeatureStore(VARIANT, device)
    rows = store.active_rows(np.arange(256, dtype=np.int64))[: min(4, config.batch_size)]
    model = CoordinateOnlyModel(0, config).to(device)
    batch = store.batch(rows, {})
    output = model(batch)
    loss, _ = e214.training_objective(output, batch, config)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite = bool(torch.isfinite(loss)) and bool(gradients) and all(
        bool(torch.all(torch.isfinite(value))) for value in gradients
    )
    if not finite:
        raise FloatingPointError("E217 finite gradient check failed")
    return {
        "finite": True,
        "loss": float(loss.detach().item()),
        "batch": int(len(rows)),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def prepare(config: e214.Config, device: torch.device, force: bool) -> dict[str, object]:
    started = time.time()
    folds = build_fold_assets(force=False)
    required = (
        support.COORDINATE_PATH,
        support.MASK_PATH,
        support.SUPPORT_REPORT,
        e213.TRAIN_H,
        TRAIN_BASE_H,
        CANONICAL_BASE_H,
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing E217 inputs: {missing}")
    caches = {
        "train_base": compact_cache(
            TRAIN_BASE_H, TRAIN_BASE_PAS, TRAIN_BASE_PDP, None, True, device, config.channel_batch_size, force
        ),
        "canonical_base": compact_cache(
            CANONICAL_BASE_H, CANONICAL_BASE_PAS, CANONICAL_BASE_PDP, None, True, device, config.channel_batch_size, force
        ),
        "official": compact_cache(
            e213.TRAIN_H, OFFICIAL_PAS, OFFICIAL_PDP, OFFICIAL_POWER, False, device, config.channel_batch_size, force
        ),
    }
    truth = e214.build_truth_components(config, device, force)
    labels = e214.build_labels(VARIANT, config, force)
    lineage = e213.fold_lineage()
    if lineage["matched_fold_count"] != 124 or lineage["label_access_risk_count"] != 376:
        raise RuntimeError(f"historical lineage mismatch: {lineage}")
    identity_error = e214.zero_action_identity(config, device)
    if identity_error != 0.0:
        raise RuntimeError(f"zero-action V7 mismatch: {identity_error}")
    gradient = finite_gradient_check(config, device)
    audit = topology_audit()
    report = {
        "experiment": VERSION,
        "status": "complete",
        "config": e214.config_payload(config),
        "folds": folds,
        "support_audit": audit,
        "model_visible_inputs": ["candidate_coordinate", "candidate_mask"],
        "neural_outputs": ["candidate_mass", "eight_port_distribution", "confidence"],
        "fixed_candidate_coordinates": True,
        "caches": caches,
        "official_h_components": truth,
        "labels": labels,
        "lineage": lineage,
        "zero_action_v7_max_abs": identity_error,
        "finite_gradient_check": gradient,
        "wall_seconds": float(time.time() - started),
        "device": str(device),
        "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "exit_code": 0,
    }
    atomic_json(PREPARE_REPORT, report)
    return report


def patch_fold_report(report: dict[str, object]) -> dict[str, object]:
    lineage = report.setdefault("data_lineage", {})
    assert isinstance(lineage, dict)
    lineage["model_visible_inputs"] = ["candidate_coordinate", "candidate_mask"]
    lineage["neural_outputs"] = ["candidate_mass", "eight_port_distribution", "confidence"]
    lineage["candidate_coordinates_fixed"] = True
    lineage["energy_used_for_selection"] = False
    lineage["forbidden_propagation_fields_used"] = []
    lineage.pop("allowed_rt_fields", None)
    lineage.pop("topology_variant", None)
    atomic_json(e214.fold_report_path(VARIANT, int(report["outer"])), report)
    return report


def aggregate(config: e214.Config, device: torch.device) -> dict[str, object]:
    report = e214.aggregate_variant(VARIANT, config, device)
    fold_deltas = [float(item["delta"]) for item in report["folds"]]
    delta = float(report["candidate"]["score"] - report["baseline_v7"]["score"])
    report["success_gate"] = {
        "minimum_delta_vs_v7": 0.005,
        "minimum_positive_folds": 4,
        "actual_delta_vs_v7": delta,
        "positive_folds": int(sum(value > 0.0 for value in fold_deltas)),
    }
    report["passed"] = bool(delta >= 0.005 and sum(value > 0.0 for value in fold_deltas) >= 4)
    report["comparison"] = {
        "e214_base_seed42": 0.6590443668078942,
        "delta_vs_e214": float(report["candidate"]["score"] - 0.6590443668078942),
        "v16_local": 0.6863750036074728,
        "delta_vs_v16": float(report["candidate"]["score"] - 0.6863750036074728),
    }
    report["data_lineage"] = {
        "official_h_labels": True,
        "v7_base": True,
        "model_visible_inputs": ["candidate_coordinate", "candidate_mask"],
        "neural_outputs": ["candidate_mass", "eight_port_distribution", "confidence"],
        "candidate_coordinates_fixed": True,
        "energy_used_for_selection": False,
        "forbidden_propagation_fields_used": [],
    }
    atomic_json(e214.final_report_path(VARIANT), report)
    return report


def config_from_args(args: argparse.Namespace) -> e214.Config:
    return replace(
        e214.Config(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_layers=0,
        coordinate_weight=0.0,
        coordinate_limits=(1.0, 1.0, 1.0),
        random_state=20_260_829_217,
    )


def status(config: e214.Config) -> dict[str, object]:
    return {
        "experiment": VERSION,
        "folds_ready": CANONICAL_META.is_file() and BROAD_FOLD.is_file(),
        "support_ready": support.COORDINATE_PATH.is_file() and support.MASK_PATH.is_file(),
        "v7_full_ready": TRAIN_BASE_H.is_file(),
        "v7_canonical_ready": CANONICAL_BASE_H.is_file(),
        "prepare_ready": PREPARE_REPORT.is_file(),
        "outer_ready": [e214.fold_report_path(VARIANT, outer).is_file() for outer in range(5)],
        "aggregate_ready": e214.final_report_path(VARIANT).is_file(),
        "config": e214.config_payload(config),
    }


def main() -> None:
    args = parse_args()
    configure()
    config = config_from_args(args)
    device = torch.device(args.device)
    try:
        if args.stage == "folds":
            result = build_fold_assets(args.force)
        elif args.stage == "prepare":
            result = prepare(config, device, args.force)
        elif args.stage == "train":
            if args.outer is None:
                raise ValueError("--outer is required for train")
            result = patch_fold_report(
                e214.train_outer(VARIANT, args.outer, config, device, args.force)
            )
        elif args.stage == "aggregate":
            result = aggregate(config, device)
        else:
            result = status(config)
        print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False), flush=True)
    except BaseException as error:
        atomic_json(
            ANALYSIS_DIR / f"failure_{args.stage}_outer{args.outer}.json",
            {
                "experiment": VERSION,
                "status": "failed",
                "stage": args.stage,
                "outer": args.outer,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "exit_code": 1,
            },
        )
        raise


if __name__ == "__main__":
    main()
