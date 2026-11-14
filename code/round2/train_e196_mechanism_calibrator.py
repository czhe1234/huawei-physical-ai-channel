#!/usr/bin/env python3
"""Train and score the strict E196 mechanism-separated spectral calibrator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

import analyze_e86_surface_endpoint_v11_blend as e86
import generate_e196_mechanism_spectra as mechanism_data
import prepare_e196_strict_path_spectra as path_data
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from common.official_metrics import aggregate_statistics, point_metric_statistics_torch
from geometry_features import GEOMETRY_CACHE
from prepare_sionna_calibration_dataset import (
    BASE_CONFIG,
    SCATTER_CONFIG,
    TRUTH_PAS,
    TRUTH_PDP,
    TRUTH_POWER,
    running_path,
)
from project_paths import PATHS
from round2.data import load_positions
from train_sionna_spectral_calibrator import (
    SpectralCalibrator,
    TrainConfig,
    spectral_cosine,
    train_epochs,
    training_loss,
)
from train_v9_joint_sionna_endpoints import (
    reliability_path,
    source_scalar_matrix,
)


VERSION = "Round2-E196-mechanism-separated-spectral-calibrator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e196_mechanism_calibrator"
FOLD_DIR = ARTIFACT_DIR / "folds"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e196_mechanism_calibrator"
PROFILE_REPORT = ANALYSIS_DIR / "profile.json"
FINAL_REPORT = ANALYSIS_DIR / "fixed500_report.json"
OUTPUT_CHANNEL = ARTIFACT_DIR / "fixed500_channel.npy"

INDEPENDENT_BASE = (
    PATHS.artifacts / "round2" / "e193_propagation_memory" / "oof_v11_channel.npy"
)
INDEPENDENT_BASE_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_e193_propagation_memory"
    / "independent_v11_base.json"
)
V15_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_official_sionna_path_calibrator"
    / "fixed500_report.json"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)

MODE = "mechanism"
DOMAINS = ("pas", "pdp")
MECHANISM_NAMES = tuple(config.name for config in mechanism_data.CONFIGS)
SIMULATION_SOURCES = ("base", "scatter", *MECHANISM_NAMES)
SOURCE_NAMES = (*SIMULATION_SOURCES, "path")
POINTS = 4000
TINY = 1e-30
TARGET_SCORE = 0.700000

# Keep the V9 model capacity and optimization budget unchanged. E196 changes only
# the observable same-coordinate source bank.
CONFIG = TrainConfig(
    epochs=12,
    steps_per_epoch=300,
    pas_batch_size=512,
    pdp_batch_size=1024,
    validation_batch_size=2048,
    hidden=96,
    context_channels=32,
    blocks=6,
    residual_scale=5.0,
    learning_rate=1.0e-3,
    weight_decay=5.0e-5,
    hellinger_weight=0.08,
    residual_weight=2.0e-4,
    seed=2026081319,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("profile", "domain", "fold", "aggregate"), required=True
    )
    parser.add_argument("--outer", type=int, choices=range(5), default=0)
    parser.add_argument("--domain", choices=DOMAINS, default="pas")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def stage_paths(outer: int, domain: str) -> dict[str, Path]:
    prefix = FOLD_DIR / f"outer{outer}_{domain}"
    return {
        "screen_checkpoint": prefix.with_name(prefix.name + "_screen.pt"),
        "screen_report": ANALYSIS_DIR / f"outer{outer}_{domain}_screen.json",
        "refit_checkpoint": prefix.with_name(prefix.name + "_refit.pt"),
        "endpoint": prefix.with_name(prefix.name + "_endpoint_f16.npy"),
        "report": ANALYSIS_DIR / f"outer{outer}_{domain}_report.json",
        "failure": ANALYSIS_DIR / f"outer{outer}_{domain}_failure.json",
    }


def fold_paths(outer: int) -> dict[str, Path]:
    return {
        "channel": FOLD_DIR / f"outer{outer}_channel.npy",
        "report": ANALYSIS_DIR / f"outer{outer}_report.json",
        "failure": ANALYSIS_DIR / f"outer{outer}_failure.json",
    }


def simulation_path(source: str, domain: str) -> Path:
    if source == "base":
        return running_path(BASE_CONFIG, "train", domain, "mean")
    if source == "scatter":
        return running_path(SCATTER_CONFIG, "train", domain, "mean")
    return completed_mechanism_path(source, domain)


def simulation_ad_path(source: str) -> Path:
    if source == "base":
        return running_path(BASE_CONFIG, "train", "ad", "mean")
    if source == "scatter":
        return running_path(SCATTER_CONFIG, "train", "ad", "mean")
    return completed_mechanism_path(source, "ad")


def mechanism_scalar_path(source: str, name: str) -> Path:
    return completed_mechanism_path(source, name)


def completed_mechanism_path(source: str, name: str) -> Path:
    root = mechanism_data.ARTIFACT_DIR / source / "full"
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    count = int(state["splits"]["train"]["count"])
    if count <= 0:
        raise RuntimeError(f"E196 mechanism source has no completed seeds: {source}")
    return mechanism_data.buffer_path(root, "train", name, "mean", count)


def path_source_paths(outer: int, mode: str) -> dict[str, Path]:
    return path_data.output_paths(outer, mode)


def required_inputs() -> tuple[Path, ...]:
    values = [
        GEOMETRY_CACHE,
        TRUTH_PAS,
        TRUTH_PDP,
        TRUTH_POWER,
        reliability_path("pas", "train"),
        reliability_path("pdp", "train"),
        FOLD_CACHE,
        OOF_META,
        INDEPENDENT_BASE,
        INDEPENDENT_BASE_REPORT,
        V15_REPORT,
        TRAIN_CHANNEL,
    ]
    for source in SIMULATION_SOURCES:
        values.extend(
            [
                simulation_path(source, "pas"),
                simulation_path(source, "pdp"),
                simulation_ad_path(source),
            ]
        )
        if source in MECHANISM_NAMES:
            values.extend(
                [
                    mechanism_scalar_path(source, "log_power"),
                    mechanism_scalar_path(source, "path_count"),
                ]
            )
    for outer in range(5):
        for mode in ("screen", "refit"):
            paths = path_source_paths(outer, mode)
            values.extend(paths[name] for name in ("pas", "pdp", "ad", "path_count"))
    return tuple(values)


def validate_lineage() -> dict[str, Any]:
    missing = [PATHS.relative(path) for path in required_inputs() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing E196 inputs: {missing}")
    report = json.loads(INDEPENDENT_BASE_REPORT.read_text(encoding="utf-8"))
    lineage = report.get("data_lineage", {})
    if lineage.get("historical_complete_channel_read") is not False:
        raise RuntimeError("E196 independent base has invalid lineage")
    base = np.load(INDEPENDENT_BASE, mmap_mode="r")
    if base.shape != (500, 256, 4, 192) or base.dtype != np.complex64:
        raise ValueError(f"invalid independent base: {base.shape}/{base.dtype}")
    return report


def normalize_probability(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), 0.0)


def tensor_from_npy(
    path: Path, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    print(f"load {PATHS.relative(path)}", flush=True)
    source = np.load(path, mmap_mode="r")
    return torch.as_tensor(source).to(device=device, dtype=dtype)


def mechanism_context() -> np.ndarray:
    columns = []
    for source in MECHANISM_NAMES:
        log_power = np.asarray(
            np.load(mechanism_scalar_path(source, "log_power"), mmap_mode="r"),
            dtype=np.float32,
        )
        path_count = np.asarray(
            np.load(mechanism_scalar_path(source, "path_count"), mmap_mode="r"),
            dtype=np.float32,
        )
        columns.extend([log_power, np.log1p(np.maximum(path_count, 0.0))])
    return np.column_stack(columns).astype(np.float32)


def shifted_ad_marginal(
    path: Path,
    site: torch.Tensor,
    domain: str,
    device: torch.device,
    *,
    already_aligned: bool,
) -> torch.Tensor:
    value = tensor_from_npy(path, device, torch.float32)
    if domain == "pas":
        result = value.sum(dim=3, dtype=torch.float32)
        if not already_aligned:
            for site_value, shift in ((0, 2), (1, 1)):
                mask = site == site_value
                result[mask] = torch.roll(result[mask], shifts=shift, dims=2)
        result = normalize_probability(result[:, None], (2, 3))
    else:
        result = value.sum(dim=(1, 2), dtype=torch.float32)
        result = normalize_probability(result[:, None], (2,))
    del value
    return result.to(torch.bfloat16)


class MechanismDomainBank:
    """GPU-resident full-resolution source bank for one official metric domain."""

    def __init__(
        self,
        domain: str,
        outer: int,
        path_mode: str,
        device: torch.device,
        validate_full: bool = False,
    ) -> None:
        if domain not in DOMAINS or path_mode not in ("screen", "refit"):
            raise ValueError((domain, path_mode))
        self.domain = domain
        self.outer = outer
        self.path_mode = path_mode
        self.device = device
        self.validate_full = validate_full
        self.source_checks: dict[str, dict[str, float | bool]] = {}
        _, _, _, train_site, _ = load_positions()
        self.site = torch.as_tensor(train_site, device=device, dtype=torch.long)
        geometry = np.load(GEOMETRY_CACHE, allow_pickle=False)["train"]
        base_context = np.concatenate(
            [
                np.asarray(geometry, dtype=np.float32),
                source_scalar_matrix(BASE_CONFIG, "train"),
                source_scalar_matrix(SCATTER_CONFIG, "train"),
                np.asarray(
                    np.load(reliability_path(domain, "train"), mmap_mode="r"),
                    dtype=np.float32,
                ),
                mechanism_context(),
            ],
            axis=1,
        ).astype(np.float32)
        self._base_context = torch.as_tensor(
            base_context, device=device, dtype=torch.float32
        )
        self.rows = len(base_context)
        power = np.load(TRUTH_POWER, mmap_mode="r")
        self.nonzero = torch.as_tensor(power > 0.0, device=device)

        self.mean: dict[str, torch.Tensor] = {}
        self.summary: dict[str, dict[str, torch.Tensor]] = {}
        self.ad: dict[str, torch.Tensor] = {}
        for source in SIMULATION_SOURCES:
            self._load_simulation_source(source)
        self._load_path_source(path_mode)

        truth_path = TRUTH_PAS if domain == "pas" else TRUTH_PDP
        truth = tensor_from_npy(truth_path, device, torch.bfloat16)
        if domain == "pas":
            self.truth = truth.permute(0, 4, 5, 1, 2, 3).reshape(
                self.rows, 768, 2, 16, 8
            ).contiguous()
            self.slice_count = 768
        else:
            self.truth = truth.reshape(self.rows, 1024, 192).contiguous()
            self.slice_count = 1024
        print(
            f"loaded E196 bank outer={outer} domain={domain} path={path_mode} "
            f"gpu_gib={torch.cuda.memory_allocated(device) / 2**30:.2f}",
            flush=True,
        )

    def _record_check(
        self, source: str, value: torch.Tensor, dims: tuple[int, ...]
    ) -> None:
        if not self.validate_full:
            return
        total = value.float().sum(dim=dims)
        normalization_error = torch.minimum(total.abs(), (total - 1.0).abs()).max()
        check = {
            "finite": bool(torch.isfinite(value).all().item()),
            "nonnegative": bool((value >= 0.0).all().item()),
            "maximum_normalization_error": float(normalization_error.item()),
        }
        if not check["finite"] or not check["nonnegative"]:
            raise RuntimeError(f"invalid E196 source {source}/{self.domain}: {check}")
        if check["maximum_normalization_error"] > 5e-3:
            raise RuntimeError(
                f"E196 normalization failed {source}/{self.domain}: {check}"
            )
        self.source_checks[source] = check

    def _load_simulation_source(self, source: str) -> None:
        value = tensor_from_npy(
            simulation_path(source, self.domain), self.device, torch.bfloat16
        ).float()
        if self.domain == "pas":
            value = value.permute(0, 4, 5, 1, 2, 3)
            value = torch.roll(value, shifts=2, dims=5)
            value = normalize_probability(value, (3, 4, 5))
            summary = {
                "ue": normalize_probability(value.sum(dim=2), (2, 3, 4)).to(
                    torch.bfloat16
                ),
                "frequency": normalize_probability(
                    value.sum(dim=1), (2, 3, 4)
                ).to(torch.bfloat16),
                "global": normalize_probability(
                    value.sum(dim=(1, 2)), (1, 2, 3)
                ).to(torch.bfloat16),
            }
        else:
            value = value.reshape(self.rows, 256, 4, 192)
            value = normalize_probability(value, (3,))
            shaped = value.reshape(self.rows, 2, 128, 4, 192)
            summary = {
                "antenna": normalize_probability(value.sum(dim=2), (2,)).to(
                    torch.bfloat16
                ),
                "ue": normalize_probability(value.sum(dim=1), (2,)).to(
                    torch.bfloat16
                ),
                "pol_ue": normalize_probability(shaped.sum(dim=2), (3,)).to(
                    torch.bfloat16
                ),
                "polarization": normalize_probability(
                    shaped.sum(dim=(2, 3)), (2,)
                ).to(torch.bfloat16),
                "global": normalize_probability(
                    value.sum(dim=(1, 2)), (1,)
                ).to(torch.bfloat16),
            }
        self.mean[source] = value.to(torch.bfloat16).contiguous()
        self._record_check(
            source,
            self.mean[source],
            (3, 4, 5) if self.domain == "pas" else (3,),
        )
        self.summary[source] = summary
        self.ad[source] = shifted_ad_marginal(
            simulation_ad_path(source),
            self.site,
            self.domain,
            self.device,
            already_aligned=False,
        )
        del value

    def _drop_path_source(self) -> None:
        self.mean.pop("path", None)
        self.summary.pop("path", None)
        self.ad.pop("path", None)

    def _load_path_source(self, mode: str) -> None:
        self._drop_path_source()
        paths = path_source_paths(self.outer, mode)
        if self.domain == "pas":
            value = tensor_from_npy(paths["pas"], self.device, torch.float32)
            value = normalize_probability(value, (2, 3, 4))
            global_value = normalize_probability(value.sum(dim=1), (1, 2, 3))
            summary = {
                "ue": value.to(torch.bfloat16),
                "frequency": global_value.to(torch.bfloat16),
                "global": global_value.to(torch.bfloat16),
            }
        else:
            value = tensor_from_npy(paths["pdp"], self.device, torch.float32)
            value = normalize_probability(value, (3,))
            summary = {
                "antenna": normalize_probability(value.sum(dim=2), (2,)).to(
                    torch.bfloat16
                ),
                "ue": normalize_probability(value.sum(dim=1), (2,)).to(
                    torch.bfloat16
                ),
                "pol_ue": value.to(torch.bfloat16),
                "polarization": normalize_probability(value.sum(dim=2), (2,)).to(
                    torch.bfloat16
                ),
                "global": normalize_probability(
                    value.sum(dim=(1, 2)), (1,)
                ).to(torch.bfloat16),
            }
        self.mean["path"] = value.to(torch.bfloat16).contiguous()
        self._record_check(
            "path",
            self.mean["path"],
            (2, 3, 4) if self.domain == "pas" else (3,),
        )
        self.summary["path"] = summary
        self.ad["path"] = shifted_ad_marginal(
            paths["ad"],
            self.site,
            self.domain,
            self.device,
            already_aligned=True,
        )
        count = tensor_from_npy(paths["path_count"], self.device, torch.float32)
        self._point_context = torch.cat(
            [self._base_context, torch.log1p(count.clamp_min(0.0))[:, None]], dim=1
        )
        self.path_mode = mode
        del value, count

    def set_path_mode(self, mode: str) -> None:
        if mode != self.path_mode:
            self._load_path_source(mode)

    def point_context(self, mode: str) -> torch.Tensor:
        if mode != MODE:
            raise ValueError(mode)
        return self._point_context

    def context_statistics(
        self, mode: str, fit_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.point_context(mode)[fit_ids]
        return value.mean(dim=0), value.std(dim=0).clamp_min(1e-4)

    def slice_context(self, slice_ids: torch.Tensor) -> torch.Tensor:
        if self.domain == "pas":
            ue = torch.div(slice_ids, 192, rounding_mode="floor")
            frequency = slice_ids % 192
            angle = (frequency.float() + 0.5) * (2.0 * math.pi / 192.0)
            return torch.cat(
                [
                    torch.nn.functional.one_hot(ue, num_classes=4).float(),
                    ((frequency.float() + 0.5) / 96.0 - 1.0)[:, None],
                    *[
                        function(harmonic * angle)[:, None]
                        for harmonic in (1.0, 2.0, 4.0, 8.0)
                        for function in (torch.sin, torch.cos)
                    ],
                ],
                dim=1,
            )
        antenna = torch.div(slice_ids, 4, rounding_mode="floor")
        ue = slice_ids % 4
        polarization = torch.div(antenna, 128, rounding_mode="floor")
        remainder = antenna % 128
        vertical = torch.div(remainder, 8, rounding_mode="floor")
        horizontal = remainder % 8
        vertical_angle = (vertical.float() + 0.5) * (2.0 * math.pi / 16.0)
        horizontal_angle = (horizontal.float() + 0.5) * (2.0 * math.pi / 8.0)
        return torch.cat(
            [
                torch.nn.functional.one_hot(polarization, num_classes=2).float(),
                torch.nn.functional.one_hot(ue, num_classes=4).float(),
                torch.sin(vertical_angle)[:, None],
                torch.cos(vertical_angle)[:, None],
                torch.sin(horizontal_angle)[:, None],
                torch.cos(horizontal_angle)[:, None],
                ((vertical.float() + 0.5) / 8.0 - 1.0)[:, None],
                ((horizontal.float() + 0.5) / 4.0 - 1.0)[:, None],
            ],
            dim=1,
        )

    @staticmethod
    def sqrt_feature(value: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(value.float().clamp_min(0.0))

    def _pas_local(
        self, source: str, point_ids: torch.Tensor, ue: torch.Tensor, frequency: torch.Tensor
    ) -> torch.Tensor:
        if source == "path":
            return self.mean[source][point_ids, ue].float()
        return self.mean[source][point_ids, ue, frequency].float()

    def _pdp_local(
        self,
        source: str,
        point_ids: torch.Tensor,
        antenna: torch.Tensor,
        ue: torch.Tensor,
        polarization: torch.Tensor,
    ) -> torch.Tensor:
        if source == "path":
            return self.mean[source][point_ids, polarization, ue].float()
        return self.mean[source][point_ids, antenna, ue].float()

    def sample_input(
        self,
        point_ids: torch.Tensor,
        slice_ids: torch.Tensor,
        mode: str,
        context_mean: torch.Tensor,
        context_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode != MODE:
            raise ValueError(mode)
        if self.domain == "pas":
            ue = torch.div(slice_ids, 192, rounding_mode="floor")
            frequency = slice_ids % 192
            local = {
                source: self._pas_local(source, point_ids, ue, frequency)
                for source in SOURCE_NAMES
            }
            maps = [local[source] for source in SOURCE_NAMES]
            maps.extend(
                (local[source] - local["base"]).abs()
                for source in SOURCE_NAMES
                if source != "base"
            )
            for source in SOURCE_NAMES:
                summary = self.summary[source]
                maps.append(summary["ue"][point_ids, ue].float())
                maps.append(
                    summary["frequency"][point_ids].float()
                    if source == "path"
                    else summary["frequency"][point_ids, frequency].float()
                )
                maps.append(summary["global"][point_ids].float())
            feature = torch.cat([self.sqrt_feature(value) for value in maps], dim=1)
            feature = torch.cat(
                [
                    feature,
                    *[
                        self.sqrt_feature(self.ad[source][point_ids])
                        for source in SOURCE_NAMES
                    ],
                ],
                dim=1,
            )
            baseline = normalize_probability(
                local["base"] + local["scatter"], (1, 2, 3)
            )
        else:
            antenna = torch.div(slice_ids, 4, rounding_mode="floor")
            ue = slice_ids % 4
            polarization = torch.div(antenna, 128, rounding_mode="floor")
            local = {
                source: self._pdp_local(
                    source, point_ids, antenna, ue, polarization
                )
                for source in SOURCE_NAMES
            }
            vectors = [local[source] for source in SOURCE_NAMES]
            vectors.extend(
                (local[source] - local["base"]).abs()
                for source in SOURCE_NAMES
                if source != "base"
            )
            for source in SOURCE_NAMES:
                summary = self.summary[source]
                vectors.extend(
                    [
                        summary["antenna"][
                            point_ids, polarization if source == "path" else antenna
                        ].float(),
                        summary["ue"][point_ids, ue].float(),
                        summary["pol_ue"][point_ids, polarization, ue].float(),
                        summary["polarization"][point_ids, polarization].float(),
                        summary["global"][point_ids].float(),
                    ]
                )
            feature = torch.cat(
                [self.sqrt_feature(value)[:, None] for value in vectors]
                + [
                    self.sqrt_feature(self.ad[source][point_ids])
                    for source in SOURCE_NAMES
                ],
                dim=1,
            )
            baseline = normalize_probability(
                local["base"] + local["scatter"], (1,)
            )
        point = (self._point_context[point_ids] - context_mean) / context_std
        context = torch.cat([point, self.slice_context(slice_ids)], dim=1)
        return feature, context, baseline

    def sample(
        self,
        point_ids: torch.Tensor,
        slice_ids: torch.Tensor,
        mode: str,
        context_mean: torch.Tensor,
        context_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature, context, baseline = self.sample_input(
            point_ids, slice_ids, mode, context_mean, context_std
        )
        truth = self.truth[point_ids, slice_ids].float()
        return feature, context, baseline, truth


def canonical_split(outer: int) -> dict[str, np.ndarray]:
    validation_fold = (outer + 1) % 5
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    outer_fit = folds[f"fold{outer}_fit"].astype(np.int64)
    validation_fit = folds[f"fold{validation_fold}_fit"].astype(np.int64)
    screen_fit = np.intersect1d(
        outer_fit, validation_fit, assume_unique=True
    ).astype(np.int64)
    outer_rows = folds[f"fold{outer}_query"].astype(np.int64)
    validation_rows = folds[f"fold{validation_fold}_query"].astype(np.int64)
    if not np.array_equal(outer_rows, query_ids[fold_id == outer]):
        raise RuntimeError(f"E196 outer{outer} canonical identity mismatch")
    if not np.array_equal(
        validation_rows, query_ids[fold_id == validation_fold]
    ):
        raise RuntimeError(
            f"E196 validation{validation_fold} canonical identity mismatch"
        )
    leakage = {
        "outer_in_screen_fit": int(np.intersect1d(outer_rows, screen_fit).size),
        "validation_in_screen_fit": int(
            np.intersect1d(validation_rows, screen_fit).size
        ),
        "outer_in_refit": int(np.intersect1d(outer_rows, outer_fit).size),
    }
    if any(leakage.values()):
        raise RuntimeError(f"E196 canonical leakage: {leakage}")
    return {
        "screen_fit": screen_fit,
        "validation": validation_rows,
        "refit": outer_fit,
        "outer": outer_rows,
        "validation_fold": np.asarray(validation_fold),
        "leakage": leakage,
    }


def endpoint_shape(domain: str, count: int) -> tuple[int, ...]:
    if domain == "pas":
        return count, 2, 16, 8, 4, 192
    return count, 256, 4, 192


def reshape_prediction(domain: str, prediction: np.ndarray) -> np.ndarray:
    if domain == "pas":
        count = len(prediction)
        return prediction.reshape(count, 4, 192, 2, 16, 8).transpose(
            0, 3, 4, 5, 1, 2
        )
    return prediction.reshape(len(prediction), 256, 4, 192)


@torch.inference_mode()
def predict_ids(
    model: SpectralCalibrator,
    bank: MechanismDomainBank,
    point_ids: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    point_ids = np.asarray(point_ids, dtype=np.int64)
    points_tensor = torch.as_tensor(point_ids, device=bank.device, dtype=torch.long)
    internal_shape = (
        (len(point_ids), bank.slice_count, 2, 16, 8)
        if bank.domain == "pas"
        else (len(point_ids), bank.slice_count, 192)
    )
    output = np.empty(internal_shape, dtype=np.float16)
    point_scores = np.empty(len(point_ids), dtype=np.float64)
    for point_start in range(0, len(point_ids), 8):
        point_end = min(point_start + 8, len(point_ids))
        local_points = points_tensor[point_start:point_end]
        points = local_points[:, None].expand(-1, bank.slice_count).reshape(-1)
        slices = torch.arange(bank.slice_count, device=bank.device).repeat(
            len(local_points)
        )
        predictions = []
        scores = []
        for start in range(0, len(points), batch_size):
            end = min(start + batch_size, len(points))
            feature, context, baseline, truth = bank.sample(
                points[start:end],
                slices[start:end],
                MODE,
                model.context_mean,
                model.context_std,
            )
            with torch.autocast(
                device_type=bank.device.type,
                dtype=torch.bfloat16,
                enabled=bank.device.type == "cuda",
            ):
                prediction, _ = model(feature, context, baseline)
            prediction = prediction.float()
            predictions.append(prediction.cpu())
            scores.append(spectral_cosine(prediction, truth).cpu())
        assembled = torch.cat(predictions, dim=0).reshape(
            len(local_points), bank.slice_count, *predictions[0].shape[1:]
        )
        output[point_start:point_end] = assembled.numpy().astype(np.float16)
        point_scores[point_start:point_end] = (
            torch.cat(scores).reshape(len(local_points), bank.slice_count)
            .mean(dim=1, dtype=torch.float64)
            .numpy()
        )
    return reshape_prediction(bank.domain, output), point_scores


def checkpoint_payload(
    model: SpectralCalibrator,
    domain: str,
    outer: int,
    path_mode: str,
    epoch: int,
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "domain": domain,
        "outer": outer,
        "path_mode": path_mode,
        "epoch": epoch,
        "mode": MODE,
        "source_names": list(SOURCE_NAMES),
        "config": asdict(CONFIG),
        "input_channels": int(model.stem.in_channels),
        "state_dict": model.state_dict(),
        "context_mean": model.context_mean.detach().cpu(),
        "context_std": model.context_std.detach().cpu(),
    }


def valid_endpoint(path: Path, domain: str, count: int) -> bool:
    if not path.is_file():
        return False
    value = np.load(path, mmap_mode="r")
    return value.shape == endpoint_shape(domain, count) and value.dtype == np.float16


def run_domain(args: argparse.Namespace) -> dict[str, Any]:
    paths = stage_paths(args.outer, args.domain)
    split = canonical_split(args.outer)
    if (
        paths["report"].is_file()
        and valid_endpoint(paths["endpoint"], args.domain, len(split["outer"]))
        and not args.force
    ):
        return json.loads(paths["report"].read_text(encoding="utf-8"))
    started = time.time()
    timing: dict[str, float] = {}
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    bank: MechanismDomainBank | None = None
    try:
        screen_cached = paths["screen_report"].is_file() and paths[
            "screen_checkpoint"
        ].is_file()
        if screen_cached and not args.force:
            screen_report = json.loads(
                paths["screen_report"].read_text(encoding="utf-8")
            )
            best_epoch = int(screen_report["best_epoch"])
        else:
            load_started = time.time()
            bank = MechanismDomainBank(
                args.domain, args.outer, "screen", device
            )
            timing["screen_resident_load"] = float(time.time() - load_started)
            train_started = time.time()
            screen_model, best_epoch, history = train_epochs(
                bank,
                MODE,
                split["screen_fit"],
                split["validation"],
                CONFIG.epochs,
                CONFIG,
                CONFIG.seed + 1000 * args.outer + 100 * DOMAINS.index(args.domain),
            )
            timing["screen_training"] = float(time.time() - train_started)
            valid = np.asarray(bank.nonzero.cpu())[split["validation"]]
            validation_rows = split["validation"][valid]
            _, validation_scores = predict_ids(
                screen_model,
                bank,
                validation_rows,
                CONFIG.validation_batch_size,
            )
            validation_cosine = float(np.mean(validation_scores))
            atomic_torch_save(
                paths["screen_checkpoint"],
                checkpoint_payload(
                    screen_model,
                    args.domain,
                    args.outer,
                    "screen",
                    best_epoch,
                ),
            )
            screen_report = {
                "experiment": VERSION,
                "status": "screen_complete",
                "outer": args.outer,
                "validation_fold": int(split["validation_fold"]),
                "domain": args.domain,
                "best_epoch": int(best_epoch),
                "validation_cosine": validation_cosine,
                "history": history,
                "split": {
                    "screen_fit": int(len(split["screen_fit"])),
                    "validation": int(len(split["validation"])),
                    "validation_nonzero": int(len(validation_rows)),
                    "outer": int(len(split["outer"])),
                    "zero_label_overlap": split["leakage"],
                },
                "checkpoint": PATHS.relative(paths["screen_checkpoint"]),
            }
            atomic_json(paths["screen_report"], screen_report)
            del screen_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if bank is None:
            load_started = time.time()
            bank = MechanismDomainBank(args.domain, args.outer, "refit", device)
            timing["refit_resident_load"] = float(time.time() - load_started)
        else:
            switch_started = time.time()
            bank.set_path_mode("refit")
            timing["path_source_switch"] = float(time.time() - switch_started)

        refit_started = time.time()
        refit_model, _, refit_history = train_epochs(
            bank,
            MODE,
            split["refit"],
            None,
            best_epoch,
            CONFIG,
            CONFIG.seed
            + 100_000
            + 1000 * args.outer
            + 100 * DOMAINS.index(args.domain),
        )
        timing["refit_training"] = float(time.time() - refit_started)
        inference_started = time.time()
        prediction, point_scores = predict_ids(
            refit_model,
            bank,
            split["outer"],
            CONFIG.validation_batch_size,
        )
        timing["outer_inference"] = float(time.time() - inference_started)
        nonzero = np.asarray(bank.nonzero.cpu())[split["outer"]]
        held_cosine = float(np.mean(point_scores[nonzero]))
        atomic_save_npy(paths["endpoint"], prediction.astype(np.float16))
        atomic_torch_save(
            paths["refit_checkpoint"],
            checkpoint_payload(
                refit_model,
                args.domain,
                args.outer,
                "refit",
                best_epoch,
            ),
        )
        timing["save"] = float(time.time() - started - sum(timing.values()))
        report = {
            "experiment": VERSION,
            "status": "domain_complete",
            "outer": args.outer,
            "validation_fold": int(split["validation_fold"]),
            "domain": args.domain,
            "sources": list(SOURCE_NAMES),
            "config": asdict(CONFIG),
            "selection": {
                "best_epoch": int(best_epoch),
                "validation_cosine": float(screen_report["validation_cosine"]),
                "held_nonzero_cosine": held_cosine,
            },
            "split": {
                "screen_fit": int(len(split["screen_fit"])),
                "validation": int(len(split["validation"])),
                "refit": int(len(split["refit"])),
                "outer": int(len(split["outer"])),
                "outer_nonzero": int(np.sum(nonzero)),
                "zero_label_overlap": split["leakage"],
            },
            "outputs": {
                "screen_checkpoint": PATHS.relative(paths["screen_checkpoint"]),
                "refit_checkpoint": PATHS.relative(paths["refit_checkpoint"]),
                "endpoint": PATHS.relative(paths["endpoint"]),
                "endpoint_shape": list(prediction.shape),
                "endpoint_dtype": str(prediction.dtype),
                "finite": bool(np.isfinite(prediction).all()),
            },
            "refit_history": refit_history,
            "data_lineage": {
                "historical_v7_v15_prediction_h_read": False,
                "v15_used_only_as_score_comparator": True,
                "independent_base": PATHS.relative(INDEPENDENT_BASE),
                "same_coordinate_mechanism_sources": list(SOURCE_NAMES),
                "outer_official_h_used_for_training": False,
                "outer_label_overlap": 0,
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
                "timing_seconds": timing,
                "measured_bottleneck": max(timing, key=timing.get),
                "wall_seconds": float(time.time() - started),
            },
        }
        atomic_json(paths["report"], report)
        return report
    except Exception:
        failure = {
            "experiment": VERSION,
            "status": "domain_failed",
            "outer": args.outer,
            "domain": args.domain,
            "traceback": traceback.format_exc(),
            "wall_seconds": float(time.time() - started),
        }
        atomic_json(paths["failure"], failure)
        raise


def metric_delta(
    selected: dict[str, float | int], baseline: dict[str, float | int]
) -> dict[str, float]:
    return {
        name: float(selected[name]) - float(baseline[name])
        for name in ("pas", "pdp", "nmse", "nmse_score", "score")
    }


def run_fold(args: argparse.Namespace) -> dict[str, Any]:
    paths = fold_paths(args.outer)
    if paths["report"].is_file() and paths["channel"].is_file() and not args.force:
        return json.loads(paths["report"].read_text(encoding="utf-8"))
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    try:
        split = canonical_split(args.outer)
        endpoint = {}
        domain_reports = {}
        for domain in DOMAINS:
            domain_paths = stage_paths(args.outer, domain)
            if not valid_endpoint(
                domain_paths["endpoint"], domain, len(split["outer"])
            ):
                raise FileNotFoundError(domain_paths["endpoint"])
            endpoint[domain] = np.asarray(
                np.load(domain_paths["endpoint"], mmap_mode="r"), dtype=np.float32
            )
            domain_reports[domain] = json.loads(
                domain_paths["report"].read_text(encoding="utf-8")
            )

        metadata = np.load(OOF_META, allow_pickle=False)
        query_ids = metadata["query_ids"].astype(np.int64)
        lookup = np.full(POINTS, -1, dtype=np.int64)
        lookup[query_ids] = np.arange(len(query_ids), dtype=np.int64)
        local = lookup[split["outer"]]
        if np.any(local < 0):
            raise RuntimeError("E196 fold contains non-canonical rows")
        base_np = np.load(INDEPENDENT_BASE, mmap_mode="r")
        truth_np = np.load(TRAIN_CHANNEL, mmap_mode="r")
        base = torch.as_tensor(
            np.asarray(base_np[local], dtype=np.complex64), device=device
        )
        truth = torch.as_tensor(
            np.asarray(truth_np[split["outer"]], dtype=np.complex64), device=device
        )
        spec = e86.Candidate(
            endpoint="e196_mechanism",
            mode="sqrt",
            pas_beta=0.20,
            pdp_beta=0.20,
            order="pas_pdp",
            cycles=4,
        )
        with torch.inference_mode():
            channel = e86.make_candidate(
                base,
                pas_distribution(base),
                pdp_distribution(base),
                torch.as_tensor(endpoint["pas"], device=device),
                torch.as_tensor(endpoint["pdp"], device=device),
                spec,
            )
            statistics = point_metric_statistics_torch(channel, truth)
        selected = aggregate_statistics(statistics)
        channel_np = channel.cpu().numpy().astype(np.complex64)
        atomic_save_npy(paths["channel"], channel_np)

        v15 = json.loads(V15_REPORT.read_text(encoding="utf-8"))
        v15_fold = next(
            row for row in v15["folds"] if int(row["outer"]) == args.outer
        )
        delta = metric_delta(selected, v15_fold["selected"])
        report = {
            "experiment": VERSION,
            "status": "fold_complete",
            "outer": args.outer,
            "count": int(len(split["outer"])),
            "selected": selected,
            "v15": v15_fold["selected"],
            "delta_vs_v15": delta,
            "policy": asdict(spec),
            "domains": {
                name: {
                    "best_epoch": domain_reports[name]["selection"]["best_epoch"],
                    "validation_cosine": domain_reports[name]["selection"][
                        "validation_cosine"
                    ],
                    "held_nonzero_cosine": domain_reports[name]["selection"][
                        "held_nonzero_cosine"
                    ],
                }
                for name in DOMAINS
            },
            "output": {
                "channel": PATHS.relative(paths["channel"]),
                "shape": list(channel_np.shape),
                "dtype": str(channel_np.dtype),
                "finite": bool(np.isfinite(channel_np).all()),
            },
            "data_lineage": {
                "historical_v15_channel_read": False,
                "independent_base": PATHS.relative(INDEPENDENT_BASE),
                "outer_official_h_used_only_for_final_scoring": True,
                "outer_label_overlap": 0,
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
                "wall_seconds": float(time.time() - started),
                "measured_bottleneck": "GPU projection and official scoring",
            },
        }
        atomic_json(paths["report"], report)
        return report
    except Exception:
        failure = {
            "experiment": VERSION,
            "status": "fold_failed",
            "outer": args.outer,
            "traceback": traceback.format_exc(),
            "wall_seconds": float(time.time() - started),
        }
        atomic_json(paths["failure"], failure)
        raise


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    validate_lineage()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    split = canonical_split(0)
    checks = {}
    for domain in DOMAINS:
        domain_started = time.time()
        bank = MechanismDomainBank(
            domain, 0, "screen", device, validate_full=True
        )
        fit = split["screen_fit"]
        fit = fit[np.asarray(bank.nonzero.cpu())[fit]]
        fit_tensor = torch.as_tensor(fit, device=device, dtype=torch.long)
        context_mean, context_std = bank.context_statistics(MODE, fit_tensor)
        batch_size = min(32, len(fit))
        point_ids = fit_tensor[:batch_size]
        slice_ids = torch.arange(batch_size, device=device) % bank.slice_count
        feature, context, baseline, truth = bank.sample(
            point_ids, slice_ids, MODE, context_mean, context_std
        )
        model = SpectralCalibrator(
            domain, feature.shape[1], context.shape[1], CONFIG
        ).to(device)
        prediction, delta = model(feature, context, baseline)
        loss = training_loss(prediction.float(), truth, delta.float(), CONFIG)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        checks[domain] = {
            "feature_shape": list(feature.shape),
            "context_shape": list(context.shape),
            "prediction_shape": list(prediction.shape),
            "input_channels": int(feature.shape[1]),
            "loss": float(loss.detach()),
            "finite": bool(
                torch.isfinite(prediction).all()
                and torch.isfinite(loss)
                and all(torch.isfinite(value).all() for value in gradients)
            ),
            "source_checks": bank.source_checks,
            "wall_seconds": float(time.time() - domain_started),
        }
        del bank, model, feature, context, baseline, truth, prediction, delta, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not all(value["finite"] for value in checks.values()):
        raise RuntimeError(f"E196 profile finite check failed: {checks}")
    report = {
        "experiment": VERSION,
        "status": "profile_complete",
        "sources": list(SOURCE_NAMES),
        "config": asdict(CONFIG),
        "split": {
            "screen_fit": int(len(split["screen_fit"])),
            "validation": int(len(split["validation"])),
            "outer": int(len(split["outer"])),
            "zero_label_overlap": split["leakage"],
        },
        "checks": checks,
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
            "wall_seconds": float(time.time() - started),
            "measured_bottleneck": "full-resolution source loading",
        },
    }
    atomic_json(PROFILE_REPORT, report)
    return report


def aggregate(device: torch.device) -> dict[str, Any]:
    started = time.time()
    validate_lineage()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    output = np.empty((500, 256, 4, 192), dtype=np.complex64)
    fold_reports = []
    for outer in range(5):
        paths = fold_paths(outer)
        if not paths["channel"].is_file() or not paths["report"].is_file():
            raise FileNotFoundError(f"missing E196 fold {outer}")
        rows = fold_id == outer
        expected_ids = query_ids[rows]
        split = canonical_split(outer)
        if not np.array_equal(expected_ids, split["outer"]):
            raise RuntimeError(f"E196 aggregate outer{outer} identity mismatch")
        channel = np.load(paths["channel"], mmap_mode="r")
        if channel.shape != (int(rows.sum()), 256, 4, 192):
            raise ValueError(f"invalid E196 fold channel {outer}: {channel.shape}")
        output[rows] = np.asarray(channel, dtype=np.complex64)
        fold_reports.append(json.loads(paths["report"].read_text(encoding="utf-8")))

    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    prediction = torch.as_tensor(output, device=device)
    truth = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device=device
    )
    statistics = point_metric_statistics_torch(prediction, truth)
    selected = aggregate_statistics(statistics)
    selected_folds = [
        {"outer": outer, **aggregate_statistics(statistics, fold_id == outer)}
        for outer in range(5)
    ]
    v15 = json.loads(V15_REPORT.read_text(encoding="utf-8"))
    v15_selected = v15["selected"]
    delta = metric_delta(selected, v15_selected)
    v15_by_outer = {int(row["outer"]): row["selected"] for row in v15["folds"]}
    fold_deltas = [
        metric_delta(row, v15_by_outer[int(row["outer"])]) for row in selected_folds
    ]
    positive_folds = int(sum(row["score"] > 0.0 for row in fold_deltas))
    worst_fold = float(min(row["score"] for row in fold_deltas))
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
        "v15": v15_selected,
        "delta_vs_v15": delta,
        "delta_vs_target": float(selected["score"] - TARGET_SCORE),
        "folds": [
            {
                "outer": outer,
                "selected": selected_folds[outer],
                "v15": v15_by_outer[outer],
                "delta_vs_v15": fold_deltas[outer],
                "pas_epoch": fold_reports[outer]["domains"]["pas"]["best_epoch"],
                "pdp_epoch": fold_reports[outer]["domains"]["pdp"]["best_epoch"],
            }
            for outer in range(5)
        ],
        "gate": {
            "target_score": TARGET_SCORE,
            "minimum_positive_folds": 4,
            "minimum_worst_fold_delta": -0.001,
            "positive_folds": positive_folds,
            "worst_fold_delta": worst_fold,
            "passed": passed,
        },
        "output": {
            "channel": PATHS.relative(OUTPUT_CHANNEL),
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(np.isfinite(output).all()),
        },
        "data_lineage": {
            "historical_v15_channel_read": False,
            "v15_used_only_as_score_comparator": True,
            "independent_base": PATHS.relative(INDEPENDENT_BASE),
            "same_coordinate_sources": list(SOURCE_NAMES),
            "official_train_h_used_for_fold_excluded_supervision": True,
            "hidden_test_h_used": False,
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
            "wall_seconds": float(time.time() - started),
            "measured_bottleneck": "batched official PAS/PDP/NMSE scoring",
        },
    }
    atomic_json(FINAL_REPORT, report)
    return report


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == "profile":
            report = run_profile(args)
        elif args.stage == "domain":
            validate_lineage()
            report = run_domain(args)
        elif args.stage == "fold":
            validate_lineage()
            report = run_fold(args)
        else:
            device = torch.device(args.device)
            torch.set_float32_matmul_precision("high")
            if device.type == "cuda":
                torch.cuda.set_device(device)
                torch.cuda.reset_peak_memory_stats()
            report = aggregate(device)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise


if __name__ == "__main__":
    main()
