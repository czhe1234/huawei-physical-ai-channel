#!/usr/bin/env python3
"""Train the E195 directional local-field path calibrator on strict holes."""

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
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

import analyze_e86_surface_endpoint_v11_blend as e86
import optimize_e98_global_array_angle_mapping as e98
import train_e190_official_sionna_path_calibrator as e190
import train_e99_corrected_array_surface_model as e99
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from common.official_metrics import aggregate_statistics, point_metric_statistics_torch
from generate_e195_directional_probe_field import (
    ANCHOR_IDS,
    ARTIFACT_DIR as PROBE_DIR,
    OFFSETS_PATH,
    PATH_SLOTS as PROBE_PATHS,
    PROBES_PER_ANCHOR,
    SEEDS,
    SITES as PROBE_SITES,
    SPLITS as PROBE_SPLITS,
    seed_paths,
)
from project_paths import PATHS
from round2.data import load_positions
from run_rbf_spectral_field import FOLD_CACHE, statistics
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E195-directional-probe-path-calibrator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e195_directional_probe_calibrator"
FIELD_DIR = ARTIFACT_DIR / "field_features"
FOLD_DIR = ARTIFACT_DIR / "folds"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e195_directional_probe_calibrator"
LOG_DIR = ANALYSIS_DIR / "logs"
REPORT = ANALYSIS_DIR / "fixed500_report.json"
OUTPUT_CHANNEL = ARTIFACT_DIR / "fixed500_channel.npy"
PROFILE_REPORT = ANALYSIS_DIR / "profile.json"

INDEPENDENT_BASE = (
    PATHS.artifacts / "round2" / "e193_propagation_memory" / "oof_v11_channel.npy"
)
INDEPENDENT_BASE_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_e193_propagation_memory"
    / "independent_v11_base.json"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
V15_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_official_sionna_path_calibrator"
    / "fixed500_report.json"
)

POINTS = 4000
CANDIDATES = e85.PATHS_PER_POINT
PROBE_FEATURES = 33
PERIODS = np.asarray((e85.MH, e85.MV, e85.S_SC), dtype=np.float32)
TINY = 1e-30
V15_SCORE = 0.6863750036074728
TARGET_SCORE = 0.700000
MAPPING_STEPS = 250
MAPPING_BATCH_SIZE = 384
MAPPING_RESTARTS = 2
MAPPING_LEARNING_RATE = 0.025


@dataclass(frozen=True)
class Config:
    outer: int = 0
    validation_fold: int = 1
    probe_hidden: int = 64
    probe_epochs: int = 40
    validation_interval: int = 4
    batch_size: int = 32
    endpoint_batch_size: int = 32
    inherited_learning_rate: float = 8e-5
    probe_learning_rate: float = 4e-4
    weight_decay: float = 2e-4
    random_state: int = 20260813195


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("profile", "fold", "aggregate"), default="fold"
    )
    parser.add_argument("--outer", type=int, choices=range(5), default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--force-field", action="store_true")
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


def atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
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
        "checkpoint": FOLD_DIR / f"outer{outer}_model.pt",
        "screen_base": FOLD_DIR / f"outer{outer}_strict_e190_screen.pt",
        "refit_base": FOLD_DIR / f"outer{outer}_strict_e190_refit.pt",
        "mapping": FOLD_DIR / f"outer{outer}_strict_mapping.npz",
        "screen_labels": FOLD_DIR / f"outer{outer}_strict_screen_labels.npz",
        "refit_labels": FOLD_DIR / f"outer{outer}_strict_refit_labels.npz",
        "channel": FOLD_DIR / f"outer{outer}_channel.npy",
        "endpoint": FOLD_DIR / f"outer{outer}_endpoint.npz",
        "report": ANALYSIS_DIR / f"outer{outer}_report.json",
    }


def field_path(outer: int, mode: str) -> Path:
    if mode not in ("screen", "refit"):
        raise ValueError(mode)
    return FIELD_DIR / f"outer{outer}_{mode}_canonical_strict_field_f16.npy"


def field_report_path(outer: int, mode: str) -> Path:
    return FIELD_DIR / f"outer{outer}_{mode}_canonical_strict_field.json"


def required_inputs() -> tuple[Path, ...]:
    probe_inputs = [
        PROBE_DIR / "state.json",
        ANCHOR_IDS,
        PROBE_SITES,
        PROBE_SPLITS,
        OFFSETS_PATH,
    ]
    for seed in SEEDS[:3]:
        paths = seed_paths(seed)
        probe_inputs.extend(
            paths[name]
            for name in ("raw", "mass", "port", "mask", "depth", "done")
        )
    return (
        *probe_inputs,
        e85.PATH_CACHE / "mask.npy",
        e85.PATH_FEATURES,
        e85.SEQUENCE_GEOMETRY,
        e85.SEQUENCE_HASH,
        e85.BASE_COORDINATE,
        e85.PORT_PRIOR,
        e85.TARGET_PAS,
        e85.TARGET_PDP,
        e85.POINT_POWER,
        e85.OOF_META,
        FOLD_CACHE,
        e190.COMPONENTS,
        INDEPENDENT_BASE,
        INDEPENDENT_BASE_REPORT,
        TRAIN_CHANNEL,
        V15_REPORT,
    )


def validate_lineage() -> dict[str, Any]:
    report = json.loads(INDEPENDENT_BASE_REPORT.read_text(encoding="utf-8"))
    lineage = report.get("data_lineage", {})
    if lineage.get("historical_complete_channel_read") is not False:
        raise RuntimeError("E195 independent base lineage is invalid")
    base = np.load(INDEPENDENT_BASE, mmap_mode="r")
    if base.shape != (500, 256, 4, 192) or base.dtype != np.complex64:
        raise ValueError(f"invalid independent base {base.shape}/{base.dtype}")
    probe_split = np.load(PROBE_SPLITS, mmap_mode="r")
    if len(probe_split) != 4500 * PROBES_PER_ANCHOR:
        raise ValueError("E195 probe split length mismatch")
    for seed in SEEDS[:3]:
        if not np.all(np.load(seed_paths(seed)["done"], mmap_mode="r")):
            raise RuntimeError(f"E195 probe seed {seed} incomplete")
    return report


def periodic_delta_torch(value: torch.Tensor, period: float) -> torch.Tensor:
    return torch.remainder(value + period / 2.0, period) - period / 2.0


def map_raw_coordinates(
    raw: torch.Tensor,
    site: torch.Tensor,
    latent: torch.Tensor,
) -> torch.Tensor:
    parameters = e98.decode_parameters(latent)
    theta = raw[..., 0]
    phi = raw[..., 1]
    yaw = parameters["yaw"][site][..., None]
    tilt = parameters["tilt"][site][..., None]
    horizontal = (
        float(e85.MH)
        * parameters["horizontal_spacing"]
        * torch.sin(theta)
        * torch.sin(phi - yaw)
        + parameters["horizontal_offset"][site][..., None]
    )
    vertical = (
        float(e85.MV)
        * parameters["vertical_spacing"]
        * (
            torch.sin(tilt) * torch.sin(theta) * torch.cos(phi - yaw)
            - torch.cos(tilt) * torch.cos(theta)
        )
        + parameters["vertical_offset"][site][..., None]
    )
    return torch.stack(
        (
            torch.remainder(horizontal, float(e85.MH)),
            torch.remainder(vertical, float(e85.MV)),
            torch.remainder(raw[..., 2], float(e85.S_SC)),
        ),
        dim=-1,
    )


def candidate_coordinates(
    latent: np.ndarray,
    device: torch.device,
    mapping_data: e98.MappingData | None = None,
    base: np.ndarray | None = None,
) -> np.ndarray:
    if mapping_data is None:
        mapping_data = e98.MappingData(device)
    if base is None:
        base = np.asarray(
            np.load(e85.BASE_COORDINATE, mmap_mode="r"), dtype=np.float32
        )
    return e190.corrected_coordinates(base, mapping_data, latent, device)


def canonical_split(
    outer: int, validation_fold: int, evaluator: "IndependentEvaluator"
) -> dict[str, np.ndarray]:
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    outer_fit = folds[f"fold{outer}_fit"].astype(np.int64)
    validation_fit = folds[f"fold{validation_fold}_fit"].astype(np.int64)
    screen_fit = np.intersect1d(
        outer_fit, validation_fit, assume_unique=True
    ).astype(np.int64)
    refit = np.asarray(outer_fit, dtype=np.int64)
    outer_rows = folds[f"fold{outer}_query"].astype(np.int64)
    validation_rows = folds[f"fold{validation_fold}_query"].astype(np.int64)
    expected_outer = evaluator.query_ids[evaluator.fold_id == outer]
    expected_validation = evaluator.query_ids[
        evaluator.fold_id == validation_fold
    ]
    if not np.array_equal(outer_rows, expected_outer):
        raise RuntimeError(f"E195 canonical outer{outer} identity mismatch")
    if not np.array_equal(validation_rows, expected_validation):
        raise RuntimeError(
            f"E195 canonical validation{validation_fold} identity mismatch"
        )
    leakage = {
        "outer_in_screen_fit": int(np.intersect1d(outer_rows, screen_fit).size),
        "validation_in_screen_fit": int(
            np.intersect1d(validation_rows, screen_fit).size
        ),
        "outer_in_refit": int(np.intersect1d(outer_rows, refit).size),
    }
    if any(leakage.values()):
        raise RuntimeError(f"E195 canonical fold leakage: {leakage}")
    return {
        "screen_fit": screen_fit,
        "refit": refit,
        "validation": validation_rows,
        "outer": outer_rows,
        "outer_ignored": folds[f"fold{outer}_ignored"].astype(np.int64),
        "validation_ignored": folds[
            f"fold{validation_fold}_ignored"
        ].astype(np.int64),
        "leakage": leakage,
    }


def load_or_fit_mapping(
    path: Path,
    screen_rows: np.ndarray,
    refit_rows: np.ndarray,
    mapping_data: e98.MappingData,
    device: torch.device,
    outer: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    screen_digest = row_digest(screen_rows.astype(np.int64))
    refit_digest = row_digest(refit_rows.astype(np.int64))
    if path.is_file():
        cached = np.load(path, allow_pickle=False)
        if (
            str(cached["screen_rows_digest"].item()) == screen_digest
            and str(cached["refit_rows_digest"].item()) == refit_digest
        ):
            return {
                "screen_latent": cached["screen_latent"].astype(np.float32),
                "refit_latent": cached["refit_latent"].astype(np.float32),
            }, {
                "status": "reused",
                "path": PATHS.relative(path),
                "screen_rows_digest": screen_digest,
                "refit_rows_digest": refit_digest,
            }

    args = SimpleNamespace(
        mapping_restarts=MAPPING_RESTARTS,
        mapping_steps=MAPPING_STEPS,
        mapping_batch_size=MAPPING_BATCH_SIZE,
        mapping_learning_rate=MAPPING_LEARNING_RATE,
    )
    started = time.time()
    screen_latent, screen_report = e99.fit_mapping(
        mapping_data,
        screen_rows,
        args,
        device,
        2026080899 + 10000 * outer,
        f"e195_outer{outer}_screen",
    )
    refit_latent, refit_report = e99.fit_mapping(
        mapping_data,
        refit_rows,
        args,
        device,
        2026081899 + 10000 * outer,
        f"e195_outer{outer}_refit",
    )
    screen_np = screen_latent.cpu().numpy().astype(np.float32)
    refit_np = refit_latent.cpu().numpy().astype(np.float32)
    atomic_save_npz(
        path,
        screen_latent=screen_np,
        refit_latent=refit_np,
        screen_rows_digest=np.asarray(screen_digest),
        refit_rows_digest=np.asarray(refit_digest),
    )
    return {
        "screen_latent": screen_np,
        "refit_latent": refit_np,
    }, {
        "status": "created",
        "path": PATHS.relative(path),
        "screen": screen_report,
        "refit": refit_report,
        "screen_rows_digest": screen_digest,
        "refit_rows_digest": refit_digest,
        "seconds": float(time.time() - started),
    }


def build_field_cache(
    outer: int,
    mode: str,
    latent: np.ndarray,
    device: torch.device,
    force: bool,
    batch_size: int = 24,
    candidate: np.ndarray | None = None,
) -> tuple[np.memmap, dict[str, Any]]:
    target = field_path(outer, mode)
    report_target = field_report_path(outer, mode)
    expected_shape = (POINTS, CANDIDATES, PROBES_PER_ANCHOR, PROBE_FEATURES)
    latent_digest = row_digest(np.asarray(latent, dtype=np.float32))
    if target.is_file() and report_target.is_file() and not force:
        cached = np.load(target, mmap_mode="r")
        cached_report = json.loads(report_target.read_text(encoding="utf-8"))
        if (
            cached.shape == expected_shape
            and cached.dtype == np.float16
            and cached_report.get("latent_digest") == latent_digest
        ):
            return cached, {**cached_report, "status": "reused"}

    started = time.time()
    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    if candidate is None:
        candidate = candidate_coordinates(latent, device)
    candidate_mask = np.asarray(
        np.load(e85.PATH_CACHE / "mask.npy", mmap_mode="r"), dtype=bool
    )
    offsets = np.asarray(np.load(OFFSETS_PATH), dtype=np.float32)
    offset_feature = np.concatenate(
        (
            offsets / 5.0,
            np.linalg.norm(offsets, axis=1, keepdims=True) / 5.0,
        ),
        axis=1,
    ).astype(np.float32)
    site_all = np.asarray(np.load(PROBE_SITES, mmap_mode="r"), dtype=np.int64)
    seed_arrays = []
    for seed in SEEDS[:3]:
        paths = seed_paths(seed)
        seed_arrays.append(
            {
                "raw": np.load(paths["raw"], mmap_mode="r"),
                "mass": np.load(paths["mass"], mmap_mode="r"),
                "port": np.load(paths["port"], mmap_mode="r"),
                "mask": np.load(paths["mask"], mmap_mode="r"),
                "depth": np.load(paths["depth"], mmap_mode="r"),
            }
        )
    latent_tensor = torch.as_tensor(latent, dtype=torch.float32, device=device)
    partial = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    output = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float16, shape=expected_shape
    )
    quality_sum = 0.0
    quality_count = 0
    for start in range(0, POINTS, batch_size):
        end = min(start + batch_size, POINTS)
        count = end - start
        probe_rows = (
            np.arange(start, end, dtype=np.int64)[:, None] * PROBES_PER_ANCHOR
            + np.arange(PROBES_PER_ANCHOR, dtype=np.int64)[None]
        )
        flat_rows = probe_rows.reshape(-1)
        probe_site = torch.as_tensor(
            site_all[flat_rows].reshape(count, PROBES_PER_ANCHOR),
            dtype=torch.long,
            device=device,
        )
        local_candidate = torch.as_tensor(
            candidate[start:end], dtype=torch.float32, device=device
        )
        seed_features = []
        for arrays in seed_arrays:
            raw = torch.as_tensor(
                np.asarray(arrays["raw"][flat_rows], dtype=np.float32).reshape(
                    count, PROBES_PER_ANCHOR, PROBE_PATHS, 3
                ),
                device=device,
            )
            coordinate = map_raw_coordinates(raw, probe_site, latent_tensor)
            valid = torch.as_tensor(
                np.asarray(arrays["mask"][flat_rows], dtype=bool).reshape(
                    count, PROBES_PER_ANCHOR, PROBE_PATHS
                ),
                device=device,
            )
            mass = torch.as_tensor(
                np.asarray(arrays["mass"][flat_rows], dtype=np.float32).reshape(
                    count, PROBES_PER_ANCHOR, PROBE_PATHS
                ),
                device=device,
            )
            port = torch.as_tensor(
                np.asarray(arrays["port"][flat_rows], dtype=np.float32).reshape(
                    count, PROBES_PER_ANCHOR, PROBE_PATHS, 8
                ),
                device=device,
            )
            depth = torch.as_tensor(
                np.asarray(arrays["depth"][flat_rows], dtype=np.float32).reshape(
                    count, PROBES_PER_ANCHOR, PROBE_PATHS
                ),
                device=device,
            )
            dh = periodic_delta_torch(
                coordinate[:, None, :, :, 0]
                - local_candidate[:, :, None, None, 0],
                float(e85.MH),
            )
            dv = periodic_delta_torch(
                coordinate[:, None, :, :, 1]
                - local_candidate[:, :, None, None, 1],
                float(e85.MV),
            )
            dd = periodic_delta_torch(
                coordinate[:, None, :, :, 2]
                - local_candidate[:, :, None, None, 2],
                float(e85.S_SC),
            )
            cost = torch.square(dh / 1.5) + torch.square(dv) + torch.square(dd / 2.0)
            cost = cost.masked_fill(~valid[:, None], 1e6)
            minimum, index = torch.min(cost, dim=3)
            batch_index = torch.arange(count, device=device)[:, None, None]
            probe_index = torch.arange(PROBES_PER_ANCHOR, device=device)[None, None, :]
            matched_coordinate = coordinate[batch_index, probe_index, index]
            matched_mass = mass[batch_index, probe_index, index]
            matched_port = port[batch_index, probe_index, index]
            matched_depth = depth[batch_index, probe_index, index]
            matched_valid = minimum < 1e5
            delta = torch.stack(
                (
                    periodic_delta_torch(
                        matched_coordinate[..., 0] - local_candidate[..., None, 0],
                        float(e85.MH),
                    )
                    / 1.5,
                    periodic_delta_torch(
                        matched_coordinate[..., 1] - local_candidate[..., None, 1],
                        float(e85.MV),
                    ),
                    periodic_delta_torch(
                        matched_coordinate[..., 2] - local_candidate[..., None, 2],
                        float(e85.S_SC),
                    )
                    / 2.0,
                ),
                dim=-1,
            )
            quality = torch.exp(-0.5 * minimum.clamp_max(40.0)) * matched_valid
            local_feature = torch.cat(
                (
                    quality[..., None],
                    torch.log1p(64.0 * matched_mass.clamp_min(0.0))[..., None],
                    delta,
                    matched_port,
                    (matched_depth / 16.0)[..., None],
                    matched_valid.to(torch.float32)[..., None],
                ),
                dim=-1,
            )
            seed_features.append(local_feature)
            quality_sum += float(quality.sum(dtype=torch.float64).item())
            quality_count += int(quality.numel())
            del raw, coordinate, valid, mass, port, depth, dh, dv, dd, cost
        stacked = torch.stack(seed_features, dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, unbiased=False)
        fixed = torch.as_tensor(offset_feature, device=device)[None, None].expand(
            count, CANDIDATES, -1, -1
        )
        feature = torch.cat((mean, std, fixed), dim=-1)
        mask = torch.as_tensor(candidate_mask[start:end], device=device)
        feature = feature * mask[:, :, None, None]
        output[start:end] = feature.cpu().numpy().astype(np.float16)
        if end % 256 < batch_size or end == POINTS:
            output.flush()
            print(
                f"E195 field outer={outer} mode={mode} {end}/{POINTS}",
                flush=True,
            )
        del stacked, mean, std, fixed, feature, seed_features
    output.flush()
    del output
    os.replace(partial, target)
    cached = np.load(target, mmap_mode="r")
    report = {
        "status": "created",
        "path": PATHS.relative(target),
        "shape": list(cached.shape),
        "dtype": str(cached.dtype),
        "latent_digest": latent_digest,
        "mean_match_quality": quality_sum / max(quality_count, 1),
        "seconds": float(time.time() - started),
    }
    atomic_json(report_target, report)
    return cached, report


class DirectionalFieldModel(e85.SurfaceInteractionModel):
    def __init__(
        self,
        base_dim: int,
        sequence_dim: int,
        model_config: e85.Config,
        probe_hidden: int,
    ) -> None:
        super().__init__(base_dim, sequence_dim, model_config, "geometry")
        hidden = model_config.hidden
        self.probe_token = nn.Sequential(
            nn.Linear(PROBE_FEATURES, probe_hidden),
            nn.GELU(),
            nn.LayerNorm(probe_hidden),
            nn.Linear(probe_hidden, probe_hidden),
            nn.GELU(),
        )
        self.probe_query = nn.Linear(hidden, probe_hidden)
        self.probe_key = nn.Linear(probe_hidden, probe_hidden)
        self.probe_value = nn.Linear(probe_hidden, probe_hidden)
        self.probe_output = nn.Sequential(
            nn.LayerNorm(3 * probe_hidden),
            nn.Linear(3 * probe_hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        nn.init.zeros_(self.probe_output[-1].weight)
        nn.init.zeros_(self.probe_output[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        base = self.base_input(batch["base"])
        sequence = self.sequence_state(
            batch["sequence_geometry"],
            batch["interactions"],
            batch["objects"],
            batch["primitives"],
            batch["depth"],
            batch["sequence_hash"],
        )
        core = self.fusion(torch.cat((base, sequence), dim=-1))
        token = self.probe_token(batch["probe"])
        query = self.probe_query(core)
        score = torch.einsum(
            "bkph,bkh->bkp", self.probe_key(token), query
        ) / math.sqrt(token.shape[-1])
        attention = torch.softmax(score, dim=2)
        value = self.probe_value(token)
        attended = torch.einsum("bkp,bkph->bkh", attention, value)
        mean = value.mean(dim=2)
        maximum = value.amax(dim=2)
        field = self.probe_output(torch.cat((attended, mean, maximum), dim=-1))
        hidden = core + field * batch["candidate_mask"][..., None]
        hidden = self.set_encoder(hidden)
        score_residual = self.config.score_limit * torch.tanh(
            self.score_head(hidden)[..., 0]
        )
        coordinate_scale = torch.tensor(
            (2.0, 4.0, 2.0), dtype=hidden.dtype, device=hidden.device
        )
        coordinate_residual = torch.tanh(self.coordinate_head(hidden)) * coordinate_scale
        port_residual = self.config.port_limit * torch.tanh(self.port_head(hidden))
        mass = torch.softmax(batch["base_log_mass"] + score_residual, dim=1)
        port = torch.softmax(batch["base_log_port"] + port_residual, dim=2).reshape(
            len(hidden), CANDIDATES, e85.MP, e85.N_UE
        )
        return {
            "mass": mass,
            "port": port,
            "coordinate": batch["coordinate"] + coordinate_residual,
            "score_residual": score_residual,
            "coordinate_residual": coordinate_residual,
            "port_residual": port_residual,
        }


class ResidentStore:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        source = e85.DataStore()
        self.base_source = source.base
        self.fallback_mass, self.fallback_port = e190.raw_prior(source)
        self.base = torch.as_tensor(
            np.asarray(source.base, dtype=np.float32), device=device
        )
        self.fraction = torch.as_tensor(
            np.asarray(source.fraction, dtype=np.float32), device=device
        )
        self.presence = torch.as_tensor(
            np.asarray(source.presence, dtype=np.float32), device=device
        )
        self.port = torch.as_tensor(
            np.asarray(source.port, dtype=np.float32).reshape(POINTS, CANDIDATES, 8),
            device=device,
        )
        self.sequence_geometry = torch.as_tensor(
            np.asarray(source.sequence_geometry), device=device
        )
        self.sequence_hash = torch.as_tensor(
            np.asarray(source.sequence_hash, dtype=np.int64), device=device
        )
        self.depth = torch.as_tensor(
            np.asarray(source.depth, dtype=np.int64), device=device
        )
        self.interactions = torch.as_tensor(
            np.asarray(source.interactions, dtype=np.int64), device=device
        )
        self.objects = torch.as_tensor(
            np.asarray(source.objects, dtype=np.int64), device=device
        )
        self.primitives = torch.as_tensor(
            np.asarray(source.primitives, dtype=np.int64), device=device
        )
        self.target_pas = torch.as_tensor(
            np.asarray(source.target_pas, dtype=np.float32), device=device
        )
        self.target_pdp = torch.as_tensor(
            np.asarray(source.target_pdp, dtype=np.float32), device=device
        )
        self.power = np.asarray(source.power, dtype=np.float64)
        self.fold = np.asarray(source.fold, dtype=np.int8)
        self.strict_ids = np.asarray(source.strict_ids, dtype=np.int64)
        self.strict_fold = np.asarray(source.strict_fold, dtype=np.int8)
        self.candidate_mask = torch.as_tensor(
            np.asarray(np.load(e85.PATH_CACHE / "mask.npy", mmap_mode="r"), dtype=bool),
            device=device,
        )
        self.coordinate = torch.empty(
            (POINTS, CANDIDATES, 3), dtype=torch.float32, device=device
        )
        self.probe = torch.empty(0, device=device)

    def normalization(self, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        total = np.zeros(self.base_source.shape[-1], dtype=np.float64)
        square = np.zeros_like(total)
        count = 0
        for start in range(0, len(rows), 128):
            values = np.asarray(
                self.base_source[rows[start : start + 128]], dtype=np.float64
            )
            total += values.sum(axis=(0, 1))
            square += np.square(values).sum(axis=(0, 1))
            count += values.shape[0] * values.shape[1]
        mean = total / max(count, 1)
        variance = np.maximum(square / max(count, 1) - mean * mean, 1e-8)
        std = np.sqrt(variance)
        std[std < 1e-4] = 1.0
        return (
            torch.as_tensor(mean.astype(np.float32), device=self.device),
            torch.as_tensor(std.astype(np.float32), device=self.device),
        )

    def set_mode(self, coordinate: np.ndarray, probe: np.ndarray) -> None:
        self.coordinate = torch.as_tensor(
            np.asarray(coordinate, dtype=np.float32), device=self.device
        )
        self.probe = torch.as_tensor(np.asarray(probe), device=self.device)

    def batch(
        self,
        rows: np.ndarray,
        mean: torch.Tensor,
        std: torch.Tensor,
        include_probe: bool = True,
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
            "candidate_mask": self.candidate_mask[ids].to(torch.float32),
        }
        if include_probe:
            if self.probe.numel() == 0:
                raise RuntimeError("E195 probe field is not loaded")
            batch["probe"] = self.probe[ids].float()
        return batch, self.target_pas[ids], self.target_pdp[ids]


class ResidentLabels:
    def __init__(self, source: Path | e190.PathLabels, device: torch.device) -> None:
        if isinstance(source, Path):
            cached = np.load(source, allow_pickle=False)
            mass = cached["mass"].astype(np.float32)
            delta = cached["delta"].astype(np.float32)
            port = cached["port"].astype(np.float32)
            valid = cached["valid"].astype(bool)
        else:
            mass = source.mass.astype(np.float32)
            delta = source.delta.astype(np.float32)
            port = source.port.astype(np.float32)
            valid = source.valid.astype(bool)
        self.mass = torch.as_tensor(mass, device=device)
        self.delta = torch.as_tensor(delta, device=device)
        self.port = torch.as_tensor(port, device=device)
        self.valid = valid


def load_or_build_labels(
    path: Path,
    coordinate: np.ndarray,
    coordinate_digest: str,
    store: ResidentStore,
    truth_coordinate: np.ndarray,
    truth_coefficient: np.ndarray,
    point_power: np.ndarray,
    config: e190.Config,
    device: torch.device,
) -> tuple[ResidentLabels, dict[str, Any]]:
    if path.is_file():
        cached = np.load(path, allow_pickle=False)
        if (
            "coordinate_digest" in cached.files
            and str(cached["coordinate_digest"].item()) == coordinate_digest
        ):
            report = (
                json.loads(str(cached["report_json"].item()))
                if "report_json" in cached.files
                else {}
            )
            return ResidentLabels(path, device), {
                "status": "reused",
                "path": PATHS.relative(path),
                "coordinate_digest": coordinate_digest,
                **report,
            }

    labels, report = e190.build_labels(
        coordinate,
        store.candidate_mask.cpu().numpy().astype(bool),
        truth_coordinate,
        truth_coefficient,
        point_power,
        store.fallback_mass,
        store.fallback_port,
        config,
    )
    atomic_save_npz(
        path,
        mass=labels.mass,
        delta=labels.delta,
        port=labels.port,
        matched_count=labels.matched_count,
        matched_fraction=labels.matched_fraction,
        valid=labels.valid,
        coordinate_digest=np.asarray(coordinate_digest),
        report_json=np.asarray(json.dumps(report, ensure_ascii=False)),
    )
    return ResidentLabels(labels, device), {
        "status": "created",
        "path": PATHS.relative(path),
        "coordinate_digest": coordinate_digest,
        **report,
    }


class IndependentEvaluator:
    def __init__(self, device: torch.device) -> None:
        metadata = np.load(e85.OOF_META, allow_pickle=False)
        self.query_ids = metadata["query_ids"].astype(np.int64)
        self.fold_id = metadata["fold_id"].astype(np.int8)
        self.lookup = np.full(POINTS, -1, dtype=np.int64)
        self.lookup[self.query_ids] = np.arange(len(self.query_ids), dtype=np.int64)
        self.base = torch.as_tensor(
            np.array(np.load(INDEPENDENT_BASE, mmap_mode="r"), copy=True), device=device
        )
        truth = np.load(TRAIN_CHANNEL, mmap_mode="r")
        self.truth = torch.as_tensor(
            np.asarray(truth[self.query_ids], dtype=np.complex64), device=device
        )
        self.truth_pas = pas_distribution(self.truth)
        self.truth_pdp = pdp_distribution(self.truth)
        self.device = device

    def evaluate(
        self,
        global_rows: np.ndarray,
        endpoint_pas: np.ndarray,
        endpoint_pdp: np.ndarray,
        collect_channel: bool = False,
    ) -> tuple[dict[str, float | int], np.ndarray | None]:
        local = self.lookup[global_rows]
        if np.any(local < 0):
            raise ValueError("E195 evaluator received non-canonical rows")
        base = self.base[local]
        spec = e86.Candidate(
            endpoint="e195_directional_probe",
            mode="sqrt",
            pas_beta=0.20,
            pdp_beta=0.20,
            order="pas_pdp",
            cycles=4,
        )
        with torch.inference_mode():
            candidate = e86.make_candidate(
                base,
                pas_distribution(base),
                pdp_distribution(base),
                torch.as_tensor(endpoint_pas, device=self.device),
                torch.as_tensor(endpoint_pdp, device=self.device),
                spec,
            )
            metric = point_metrics(
                candidate,
                self.truth[local],
                self.truth_pas[local],
                self.truth_pdp[local],
            )
        result = statistics(*metric)
        channel = candidate.cpu().numpy().astype(np.complex64) if collect_channel else None
        return result, channel

    def score_channel(self, channel: np.ndarray) -> dict[str, float | int]:
        prediction = torch.as_tensor(channel, device=self.device)
        metric = point_metrics(
            prediction, self.truth, self.truth_pas, self.truth_pdp
        )
        return statistics(*metric)


def model_config_from_e190(config: e190.Config) -> e85.Config:
    return e85.Config(
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


def model_config(checkpoint: dict[str, Any]) -> e85.Config:
    return model_config_from_e190(e190.Config(**checkpoint["config"]))


def make_model(
    checkpoint: dict[str, Any],
    store: ResidentStore,
    config: Config,
    device: torch.device,
    seed: int,
) -> tuple[DirectionalFieldModel, dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = DirectionalFieldModel(
        store.base.shape[-1],
        store.sequence_geometry.shape[-1],
        model_config(checkpoint),
        config.probe_hidden,
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if unexpected or any(not name.startswith("probe_") for name in missing):
        raise RuntimeError(
            f"E195 checkpoint load mismatch missing={missing} unexpected={unexpected}"
        )
    return model, {"missing": list(missing), "unexpected": list(unexpected)}


@torch.inference_mode()
def identity_check(
    checkpoint: dict[str, Any],
    model: DirectionalFieldModel,
    store: ResidentStore,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> float:
    baseline = e85.SurfaceInteractionModel(
        store.base.shape[-1],
        store.sequence_geometry.shape[-1],
        model_config(checkpoint),
        "geometry",
    ).to(device)
    baseline.load_state_dict(checkpoint["state_dict"])
    baseline.eval()
    model.eval()
    batch, _, _ = store.batch(rows, mean, std)
    left = baseline(batch)
    right = model(batch)
    error = max(
        float(torch.max(torch.abs(left[name] - right[name])).item())
        for name in ("mass", "port", "coordinate")
    )
    del baseline
    return error


def endpoint_arrays(
    model: nn.Module,
    store: ResidentStore,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    include_probe: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    mass = np.empty((len(rows), CANDIDATES), dtype=np.float32)
    coordinate = np.empty((len(rows), CANDIDATES, 3), dtype=np.float32)
    port = np.empty((len(rows), CANDIDATES, e85.MP, e85.N_UE), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch, _, _ = store.batch(
                rows[start:end], mean, std, include_probe=include_probe
            )
            output = model(batch)
            mass[start:end] = output["mass"].cpu().numpy().astype(np.float32)
            coordinate[start:end] = (
                output["coordinate"].cpu().numpy() % PERIODS
            ).astype(np.float32)
            port[start:end] = output["port"].cpu().numpy().astype(np.float32)
    return mass, coordinate, port


def evaluate_model(
    model: DirectionalFieldModel,
    store: ResidentStore,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    evaluator: IndependentEvaluator,
    config: Config,
    collect_channel: bool = False,
) -> tuple[dict[str, float | int], dict[str, np.ndarray], np.ndarray | None]:
    mass, coordinate, port = endpoint_arrays(
        model, store, rows, mean, std, config.endpoint_batch_size
    )
    endpoint_pas, endpoint_pdp = e190.render_endpoint(
        mass, coordinate, port, evaluator.device
    )
    metrics, channel = evaluator.evaluate(
        rows, endpoint_pas, endpoint_pdp, collect_channel=collect_channel
    )
    return metrics, {"mass": mass, "coordinate": coordinate, "port": port}, channel


def make_fresh_base_model(
    store: ResidentStore,
    config: e190.Config,
    device: torch.device,
    seed: int,
) -> e85.SurfaceInteractionModel:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return e85.SurfaceInteractionModel(
        store.base.shape[-1],
        store.sequence_geometry.shape[-1],
        model_config_from_e190(config),
        "geometry",
    ).to(device)


def evaluate_base_model(
    model: e85.SurfaceInteractionModel,
    store: ResidentStore,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    evaluator: IndependentEvaluator,
    endpoint_batch_size: int,
) -> dict[str, float | int]:
    mass, coordinate, port = endpoint_arrays(
        model,
        store,
        rows,
        mean,
        std,
        endpoint_batch_size,
        include_probe=False,
    )
    endpoint_pas, endpoint_pdp = e190.render_endpoint(
        mass, coordinate, port, evaluator.device
    )
    metrics, _ = evaluator.evaluate(rows, endpoint_pas, endpoint_pdp)
    return metrics


def train_base_epoch(
    model: e85.SurfaceInteractionModel,
    optimizer: torch.optim.Optimizer,
    store: ResidentStore,
    labels: ResidentLabels,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    config: e190.Config,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    eligible = rows[labels.valid[rows] & (store.power[rows] > 0.0)]
    order = rng.permutation(eligible)
    totals: dict[str, list[float]] = {
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
    for start in range(0, len(order), config.batch_size):
        local = order[start : start + config.batch_size]
        ids = torch.as_tensor(local, dtype=torch.long, device=store.device)
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
            config,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError("non-finite strict E190 gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        totals["loss"].append(float(loss.detach().item()))
        for name, value in values.items():
            totals[name].append(float(value.detach().item()))
    return {name: float(np.mean(values)) for name, values in totals.items()}


def train_strict_base_screen(
    path: Path,
    store: ResidentStore,
    labels: ResidentLabels,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    evaluator: IndependentEvaluator,
    config: e190.Config,
    endpoint_batch_size: int,
    split_digest: str,
    coordinate_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_file():
        checkpoint = torch.load(path, map_location=store.device, weights_only=False)
        if (
            checkpoint.get("split_digest") == split_digest
            and checkpoint.get("coordinate_digest") == coordinate_digest
        ):
            return checkpoint, {
                "status": "reused",
                "path": PATHS.relative(path),
                "best_epoch": int(checkpoint["best_epoch"]),
            }

    started = time.time()
    model = make_fresh_base_model(store, config, store.device, config.random_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
        eta_min=config.learning_rate * 0.08,
    )
    rng = np.random.default_rng(config.random_state)
    best_epoch = 0
    best_score = -math.inf
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        train = train_base_epoch(
            model, optimizer, store, labels, fit_rows, mean, std, config, rng
        )
        scheduler.step()
        record: dict[str, Any] = {"epoch": epoch, "train": train}
        if epoch >= config.minimum_epochs and epoch % config.validation_interval == 0:
            validation = evaluate_base_model(
                model,
                store,
                validation_rows,
                mean,
                std,
                evaluator,
                endpoint_batch_size,
            )
            record["validation"] = validation
            if float(validation["score"]) > best_score:
                best_score = float(validation["score"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"E195 strict E190 outer={config.outer} epoch={epoch}/{config.epochs} "
                f"validation={float(validation['score']):.6f} best={best_score:.6f}",
                flush=True,
            )
        history.append(record)
    if best_state is None or best_epoch == 0:
        raise RuntimeError(f"strict E190 outer{config.outer} selected no epoch")
    checkpoint = {
        "state_dict": best_state,
        "config": asdict(config),
        "feature_mean": mean.cpu().numpy(),
        "feature_std": std.cpu().numpy(),
        "best_epoch": best_epoch,
        "outer": config.outer,
        "validation_fold": config.validation_fold,
        "split_digest": split_digest,
        "coordinate_digest": coordinate_digest,
        "history": history,
    }
    atomic_torch_save(path, checkpoint)
    del model
    return checkpoint, {
        "status": "created",
        "path": PATHS.relative(path),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "seconds": float(time.time() - started),
    }


def train_strict_base_refit(
    path: Path,
    store: ResidentStore,
    labels: ResidentLabels,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    epochs: int,
    config: e190.Config,
    split_digest: str,
    coordinate_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_file():
        checkpoint = torch.load(path, map_location=store.device, weights_only=False)
        if (
            checkpoint.get("split_digest") == split_digest
            and checkpoint.get("coordinate_digest") == coordinate_digest
            and int(checkpoint.get("best_epoch", -1)) == epochs
        ):
            return checkpoint, {
                "status": "reused",
                "path": PATHS.relative(path),
                "epochs": epochs,
            }

    started = time.time()
    model = make_fresh_base_model(
        store, config, store.device, config.random_state + 100000
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=config.learning_rate * 0.08
    )
    rng = np.random.default_rng(config.random_state + 100000)
    history = []
    for epoch in range(1, epochs + 1):
        values = train_base_epoch(
            model, optimizer, store, labels, rows, mean, std, config, rng
        )
        scheduler.step()
        history.append({"epoch": epoch, **values})
        if epoch % config.validation_interval == 0 or epoch == epochs:
            print(
                f"E195 strict E190 refit outer={config.outer} "
                f"epoch={epoch}/{epochs} mass={values['mass_affinity']:.6f}",
                flush=True,
            )
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "feature_mean": mean.cpu().numpy(),
        "feature_std": std.cpu().numpy(),
        "best_epoch": epochs,
        "outer": config.outer,
        "validation_fold": config.validation_fold,
        "split_digest": split_digest,
        "coordinate_digest": coordinate_digest,
        "history": history,
    }
    atomic_torch_save(path, checkpoint)
    del model
    return checkpoint, {
        "status": "created",
        "path": PATHS.relative(path),
        "epochs": epochs,
        "seconds": float(time.time() - started),
    }


def train_epoch(
    model: DirectionalFieldModel,
    optimizer: torch.optim.Optimizer,
    store: ResidentStore,
    labels: ResidentLabels,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    config: Config,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    eligible = rows[labels.valid[rows] & (store.power[rows] > 0.0)]
    order = rng.permutation(eligible)
    totals: dict[str, list[float]] = {
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
    for start in range(0, len(order), config.batch_size):
        local = order[start : start + config.batch_size]
        ids = torch.as_tensor(local, dtype=torch.long, device=store.device)
        batch, target_pas, target_pdp = store.batch(local, mean, std)
        output = model(batch)
        loss, values = e190.label_objective(
            output,
            batch,
            target_pas,
            target_pdp,
            labels.mass[ids],
            labels.delta[ids],
            labels.port[ids],
            e190.Config(**{"outer": config.outer, "validation_fold": config.validation_fold}),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError("non-finite E195 gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        totals["loss"].append(float(loss.detach().item()))
        for name, value in values.items():
            totals[name].append(float(value.detach().item()))
    return {name: float(np.mean(value)) for name, value in totals.items()}


def optimizer_for(
    model: DirectionalFieldModel, config: Config, epochs: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    probe_parameters = []
    inherited_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith("probe_"):
            probe_parameters.append(parameter)
        else:
            inherited_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        (
            {
                "params": inherited_parameters,
                "lr": config.inherited_learning_rate,
            },
            {"params": probe_parameters, "lr": config.probe_learning_rate},
        ),
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=config.inherited_learning_rate * 0.08
    )
    return optimizer, scheduler


def train_screen(
    model: DirectionalFieldModel,
    store: ResidentStore,
    labels: ResidentLabels,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    evaluator: IndependentEvaluator,
    config: Config,
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    baseline, _, _ = evaluate_model(
        model, store, validation_rows, mean, std, evaluator, config
    )
    best_score = float(baseline["score"])
    best_epoch = 0
    best_state: dict[str, Any] = copy.deepcopy(model.state_dict())
    history: list[dict[str, Any]] = [
        {"epoch": 0, "validation": baseline, "fallback": "fresh E190 parameters"}
    ]
    optimizer, scheduler = optimizer_for(model, config, config.probe_epochs)
    rng = np.random.default_rng(config.random_state + config.outer)
    for epoch in range(1, config.probe_epochs + 1):
        train = train_epoch(
            model, optimizer, store, labels, fit_rows, mean, std, config, rng
        )
        scheduler.step()
        record: dict[str, Any] = {"epoch": epoch, "train": train}
        if epoch % config.validation_interval == 0:
            validation, _, _ = evaluate_model(
                model, store, validation_rows, mean, std, evaluator, config
            )
            record["validation"] = validation
            record["delta_vs_epoch0"] = e190.metric_delta(validation, baseline)
            if float(validation["score"]) > best_score:
                best_score = float(validation["score"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"E195 outer={config.outer} epoch={epoch}/{config.probe_epochs} "
                f"train_pas={train['pas']:.6f} train_pdp={train['pdp']:.6f} "
                f"validation={float(validation['score']):.6f} best={best_score:.6f}",
                flush=True,
            )
        history.append(record)
    return best_epoch, best_state, history


def train_refit(
    model: DirectionalFieldModel,
    store: ResidentStore,
    labels: ResidentLabels,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    epochs: int,
    config: Config,
) -> list[dict[str, float]]:
    if epochs == 0:
        return []
    optimizer, scheduler = optimizer_for(model, config, epochs)
    rng = np.random.default_rng(config.random_state + 100_000 + config.outer)
    history = []
    for epoch in range(1, epochs + 1):
        values = train_epoch(
            model, optimizer, store, labels, rows, mean, std, config, rng
        )
        scheduler.step()
        history.append({"epoch": epoch, **values})
        if epoch % config.validation_interval == 0 or epoch == epochs:
            print(
                f"E195 refit outer={config.outer} epoch={epoch}/{epochs} "
                f"mass={values['mass_affinity']:.6f} "
                f"pas={values['pas']:.6f} pdp={values['pdp']:.6f}",
                flush=True,
            )
    return history


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
    base_config = e190.Config(
        outer=config.outer, validation_fold=config.validation_fold
    )
    lineage = validate_lineage()

    load_start = time.time()
    store = ResidentStore(device)
    evaluator = IndependentEvaluator(device)
    split = canonical_split(config.outer, config.validation_fold, evaluator)
    screen_fit = split["screen_fit"]
    refit_rows = split["refit"]
    validation_rows = split["validation"]
    outer_rows = split["outer"]
    mapping_data = e98.MappingData(device)
    base_coordinate = np.asarray(
        np.load(e85.BASE_COORDINATE, mmap_mode="r"), dtype=np.float32
    )
    component = np.load(e190.COMPONENTS, allow_pickle=False)
    truth_coordinate = component["coordinates"].astype(np.float32)
    truth_coefficient = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)
    phase["resident_load"] = float(time.time() - load_start)

    mapping_start = time.time()
    mapping, mapping_report = load_or_fit_mapping(
        paths["mapping"],
        screen_fit,
        refit_rows,
        mapping_data,
        device,
        config.outer,
    )
    phase["strict_mapping"] = float(time.time() - mapping_start)

    field_start = time.time()
    screen_coordinate = candidate_coordinates(
        mapping["screen_latent"], device, mapping_data, base_coordinate
    )
    screen_coordinate_digest = row_digest(
        mapping["screen_latent"], screen_coordinate
    )
    screen_labels, screen_label_report = load_or_build_labels(
        paths["screen_labels"],
        screen_coordinate,
        screen_coordinate_digest,
        store,
        truth_coordinate,
        truth_coefficient,
        point_power,
        base_config,
        device,
    )
    screen_field, screen_field_report = build_field_cache(
        config.outer,
        "screen",
        mapping["screen_latent"],
        device,
        args.force_field,
        candidate=screen_coordinate,
    )
    store.set_mode(screen_coordinate, screen_field)
    phase["screen_field"] = float(time.time() - field_start)

    mean_screen, std_screen = store.normalization(screen_fit)
    screen_split_digest = row_digest(
        screen_fit, validation_rows, outer_rows
    )
    base_screen_start = time.time()
    screen_checkpoint, base_screen_report = train_strict_base_screen(
        paths["screen_base"],
        store,
        screen_labels,
        screen_fit,
        validation_rows,
        mean_screen,
        std_screen,
        evaluator,
        base_config,
        config.endpoint_batch_size,
        screen_split_digest,
        screen_coordinate_digest,
    )
    phase["strict_e190_screen"] = float(time.time() - base_screen_start)
    base_best_epoch = int(screen_checkpoint["best_epoch"])
    model, load_report = make_model(
        screen_checkpoint,
        store,
        config,
        device,
        config.random_state + config.outer,
    )
    identity_error = identity_check(
        screen_checkpoint,
        model,
        store,
        screen_fit[: min(4, len(screen_fit))],
        mean_screen,
        std_screen,
        device,
    )
    if identity_error > 1e-6:
        raise RuntimeError(f"E195 epoch0 identity failed: {identity_error}")

    train_start = time.time()
    best_epoch, _, screen_history = train_screen(
        model,
        store,
        screen_labels,
        screen_fit,
        validation_rows,
        mean_screen,
        std_screen,
        evaluator,
        config,
    )
    phase["screen_training"] = float(time.time() - train_start)
    del model, screen_labels, screen_field, screen_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    refit_field_start = time.time()
    refit_coordinate = candidate_coordinates(
        mapping["refit_latent"], device, mapping_data, base_coordinate
    )
    refit_coordinate_digest = row_digest(
        mapping["refit_latent"], refit_coordinate
    )
    refit_labels, refit_label_report = load_or_build_labels(
        paths["refit_labels"],
        refit_coordinate,
        refit_coordinate_digest,
        store,
        truth_coordinate,
        truth_coefficient,
        point_power,
        base_config,
        device,
    )
    refit_field, refit_field_report = build_field_cache(
        config.outer,
        "refit",
        mapping["refit_latent"],
        device,
        args.force_field,
        candidate=refit_coordinate,
    )
    store.set_mode(refit_coordinate, refit_field)
    phase["refit_field"] = float(time.time() - refit_field_start)
    mean_refit, std_refit = store.normalization(refit_rows)
    refit_split_digest = row_digest(refit_rows, outer_rows)
    base_refit_start = time.time()
    refit_checkpoint, base_refit_report = train_strict_base_refit(
        paths["refit_base"],
        store,
        refit_labels,
        refit_rows,
        mean_refit,
        std_refit,
        base_best_epoch,
        base_config,
        refit_split_digest,
        refit_coordinate_digest,
    )
    phase["strict_e190_refit"] = float(time.time() - base_refit_start)
    model, refit_load_report = make_model(
        refit_checkpoint,
        store,
        config,
        device,
        config.random_state + 100_000 + config.outer,
    )
    refit_identity = identity_check(
        refit_checkpoint,
        model,
        store,
        refit_rows[: min(4, len(refit_rows))],
        mean_refit,
        std_refit,
        device,
    )
    if refit_identity > 1e-6:
        raise RuntimeError(f"E195 refit identity failed: {refit_identity}")
    refit_start = time.time()
    refit_history = train_refit(
        model,
        store,
        refit_labels,
        refit_rows,
        mean_refit,
        std_refit,
        best_epoch,
        config,
    )
    selected, endpoint, channel = evaluate_model(
        model,
        store,
        outer_rows,
        mean_refit,
        std_refit,
        evaluator,
        config,
        collect_channel=True,
    )
    phase["refit_training_and_outer"] = float(time.time() - refit_start)
    if channel is None:
        raise RuntimeError("E195 outer channel missing")

    v15 = json.loads(V15_REPORT.read_text(encoding="utf-8"))
    v15_fold = next(row for row in v15["folds"] if int(row["outer"]) == config.outer)
    delta = e190.metric_delta(selected, v15_fold["selected"])
    atomic_save_npy(paths["channel"], channel)
    temporary_endpoint = paths["endpoint"].with_name(
        f".{paths['endpoint'].name}.{os.getpid()}.tmp.npz"
    )
    np.savez_compressed(
        temporary_endpoint,
        query_ids=outer_rows.astype(np.int32),
        mass=endpoint["mass"],
        coordinate=endpoint["coordinate"],
        port=endpoint["port"],
    )
    os.replace(temporary_endpoint, paths["endpoint"])
    atomic_torch_save(
        paths["checkpoint"],
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "feature_mean": mean_refit.cpu().numpy(),
            "feature_std": std_refit.cpu().numpy(),
            "best_probe_epoch": best_epoch,
            "best_base_epoch": int(refit_checkpoint["best_epoch"]),
            "outer": config.outer,
            "mapping_latent": mapping["refit_latent"],
            "split_digest": refit_split_digest,
        },
    )
    phase["save"] = float(time.time() - started - sum(phase.values()))

    peak_memory = (
        float(torch.cuda.max_memory_allocated(device) / 2**30)
        if device.type == "cuda"
        else 0.0
    )
    report = {
        "experiment": VERSION,
        "status": "fold_complete",
        "configuration": asdict(config),
        "outer": config.outer,
        "validation_fold": config.validation_fold,
        "split": {
            "screen_fit": int(len(screen_fit)),
            "refit": int(len(refit_rows)),
            "validation": int(len(validation_rows)),
            "outer": int(len(outer_rows)),
            "outer_ignored": int(len(split["outer_ignored"])),
            "validation_ignored": int(len(split["validation_ignored"])),
            "zero_label_overlap": split["leakage"],
            "screen_fit_digest": row_digest(screen_fit),
            "refit_digest": row_digest(refit_rows),
        },
        "identity": {
            "screen_max_abs_error": identity_error,
            "refit_max_abs_error": refit_identity,
            "screen_checkpoint_load": load_report,
            "refit_checkpoint_load": refit_load_report,
        },
        "strict_initialization": {
            "mapping": mapping_report,
            "screen_labels": screen_label_report,
            "refit_labels": refit_label_report,
            "screen_model": base_screen_report,
            "refit_model": base_refit_report,
            "historical_e190_checkpoint_read": False,
        },
        "field": {
            "screen": screen_field_report,
            "refit": refit_field_report,
        },
        "selection": {
            "best_probe_epoch": best_epoch,
            "screen_history": screen_history,
            "refit_history": refit_history,
        },
        "selected": selected,
        "v15": v15_fold["selected"],
        "delta_vs_v15": delta,
        "data_lineage": {
            "historical_v7_v15_prediction_h_read": False,
            "historical_test_endpoint_read": False,
            "historical_e190_checkpoint_read": False,
            "official_train_h_used_for_labels": True,
            "directional_sionna_probes_used": True,
            "canonical_fit_cache": PATHS.relative(FOLD_CACHE),
            "independent_base_report": PATHS.relative(INDEPENDENT_BASE_REPORT),
            "independent_base_lineage": lineage.get("data_lineage", {}),
        },
        "output": {
            "channel": PATHS.relative(paths["channel"]),
            "endpoint": PATHS.relative(paths["endpoint"]),
            "checkpoint": PATHS.relative(paths["checkpoint"]),
            "shape": list(channel.shape),
            "dtype": str(channel.dtype),
            "finite": bool(np.isfinite(channel).all()),
        },
        "performance": {
            "device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
            "peak_torch_memory_gib": peak_memory,
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "timing_seconds": phase,
            "measured_bottleneck": max(phase, key=phase.get),
        },
        "seconds": float(time.time() - started),
    }
    atomic_json(paths["report"], report)
    return report


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    config = Config(outer=args.outer, validation_fold=(args.outer + 1) % 5)
    base_config = e190.Config(
        outer=config.outer, validation_fold=config.validation_fold
    )
    lineage = validate_lineage()
    paths = fold_paths(config.outer)
    store = ResidentStore(device)
    evaluator = IndependentEvaluator(device)
    split = canonical_split(config.outer, config.validation_fold, evaluator)
    mapping_data = e98.MappingData(device)
    mapping, mapping_report = load_or_fit_mapping(
        paths["mapping"],
        split["screen_fit"],
        split["refit"],
        mapping_data,
        device,
        config.outer,
    )
    base_coordinate = np.asarray(
        np.load(e85.BASE_COORDINATE, mmap_mode="r"), dtype=np.float32
    )
    coordinate = candidate_coordinates(
        mapping["screen_latent"], device, mapping_data, base_coordinate
    )
    coordinate_digest = row_digest(mapping["screen_latent"], coordinate)
    field, field_report = build_field_cache(
        config.outer,
        "screen",
        mapping["screen_latent"],
        device,
        args.force_field,
        batch_size=8,
        candidate=coordinate,
    )
    store.set_mode(coordinate, field)
    labels, label_report = load_or_build_labels(
        paths["screen_labels"],
        coordinate,
        coordinate_digest,
        store,
        np.load(e190.COMPONENTS, allow_pickle=False)["coordinates"].astype(
            np.float32
        ),
        np.load(e190.COMPONENTS, allow_pickle=False)["coefficients"].astype(
            np.complex64
        ),
        np.load(e190.COMPONENTS, allow_pickle=False)["point_power"].astype(
            np.float64
        ),
        base_config,
        device,
    )
    mean, std = store.normalization(split["screen_fit"])
    base_model = make_fresh_base_model(
        store, base_config, device, base_config.random_state
    )
    checkpoint = {
        "state_dict": base_model.state_dict(),
        "config": asdict(base_config),
    }
    model, load_report = make_model(
        checkpoint,
        store,
        config,
        device,
        config.random_state + config.outer,
    )
    sample_rows = split["screen_fit"][
        labels.valid[split["screen_fit"]]
        & (store.power[split["screen_fit"]] > 0.0)
    ][: config.batch_size]
    if len(sample_rows) == 0:
        raise RuntimeError("E195 profile found no eligible strict fit rows")
    identity_error = identity_check(
        checkpoint, model, store, sample_rows[:4], mean, std, device
    )
    if identity_error > 1e-6:
        raise RuntimeError(f"E195 profile identity failed: {identity_error}")
    optimizer, _ = optimizer_for(model, config, 1)
    gradient_values = train_epoch(
        model,
        optimizer,
        store,
        labels,
        sample_rows,
        mean,
        std,
        config,
        np.random.default_rng(config.random_state),
    )
    report = {
        "experiment": VERSION,
        "status": "profile_complete",
        "outer": config.outer,
        "validation_fold": config.validation_fold,
        "split": {
            "screen_fit": int(len(split["screen_fit"])),
            "refit": int(len(split["refit"])),
            "validation": int(len(split["validation"])),
            "outer": int(len(split["outer"])),
            "zero_label_overlap": split["leakage"],
        },
        "mapping": mapping_report,
        "field": field_report,
        "labels": label_report,
        "checks": {
            "identity_max_abs_error": identity_error,
            "one_batch_gradient": gradient_values,
        },
        "shape": list(field.shape),
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "peak_torch_memory_gib": float(torch.cuda.max_memory_allocated(device) / 2**30)
        if device.type == "cuda"
        else 0.0,
        "data_lineage": {
            "historical_e190_mapping_read": False,
            "historical_e190_checkpoint_read": False,
            "canonical_split": True,
            "independent_base_lineage": lineage.get("data_lineage", {}),
        },
        "seconds": float(time.time() - started),
    }
    atomic_json(PROFILE_REPORT, report)
    return report


def aggregate(device: torch.device) -> dict[str, Any]:
    started = time.time()
    validate_lineage()
    evaluator = IndependentEvaluator(device)
    output = np.empty((500, 256, 4, 192), dtype=np.complex64)
    fold_reports = []
    deltas = []
    for outer in range(5):
        paths = fold_paths(outer)
        if not paths["report"].is_file() or not paths["channel"].is_file():
            raise FileNotFoundError(f"missing E195 outer {outer}")
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        rows = evaluator.fold_id == outer
        local_channel = np.load(paths["channel"], mmap_mode="r")
        if local_channel.shape != (int(rows.sum()), 256, 4, 192):
            raise ValueError(f"E195 outer {outer} channel shape mismatch")
        output[rows] = np.asarray(local_channel, dtype=np.complex64)
        fold_reports.append(
            {
                "outer": outer,
                "count": int(rows.sum()),
                "selected": report["selected"],
                "v15": report["v15"],
                "delta_vs_v15": report["delta_vs_v15"],
                "best_probe_epoch": report["selection"]["best_probe_epoch"],
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
        raise RuntimeError(f"E195 official replay mismatch: {replay_error}")

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
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    missing = [PATHS.relative(path) for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing E195 inputs: {missing}")
    try:
        if args.stage == "profile":
            report = run_profile(args)
        elif args.stage == "fold":
            report = run_fold(args)
        else:
            report = aggregate(torch.device(args.device))
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
