"""Rebuild the frozen complex base field from official inputs and local physics."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import re
import resource
import time
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from .base_runtime.carrier_phase import estimate_carrier
from .base_runtime.spatial_baselines import BaselineConfig, predict_one

from .base_runtime import carrier_bank
from .base_runtime import calibrated_spectral_field as spectral_stage
from .base_runtime import path_delay_field as path_link_stage
from .base_runtime import reliable_angle_field as reliability_stage
from .base_runtime import path_link_model
from .base_runtime import joint_spectral_model as joint_stage
from .base_runtime.exact_projection import (
    ACTIONS,
    continue_trajectory,
    initialize_trajectory,
)
from .base_runtime.grouped_prediction import (
    blend_group_pas,
    blend_group_pdp,
    globally_shrunk_targets,
    predict_grouped,
)
from .base_runtime.donor_encoder import spectral_target
from .base_runtime import propagation_lifecycle as lifecycle
from .base_runtime.calibrated_exact_projection import (
    apply_calibration,
    mode_targets,
)
from .base_runtime.joint_spectral_projection import Candidate as JointCandidate
from .base_runtime.joint_spectral_projection import (
    apply_candidate as apply_joint_candidate,
)
from .config import PipelineConfig, RuntimePaths
from .data import OfficialData, load_official_data
from .geometry import build_geometry_regime
from .features import load_model_inputs
from .io_utils import atomic_json, atomic_npy, atomic_npz, validate_channel
from .path_tokens import cache_dir as path_token_dir


TINY = 1e-30
OFFICIAL_SPECTRAL_TINY = 1e-20
REFERENCE_FIELD_SCALES = {0: 0.7137276386108562, 1: 0.08192739921591057}
STACKED_FIELD_SCALES = {0: 0.9398906028761005, 1: 0.4924343744157239}
EXACT_PATTERN = re.compile(
    r"^exact_(raw|pascal|pdpcal|bothcal)_(pas_pdp|pdp_pas)_(.+)_c(\d+)$"
)
GEODESIC_NAMES = (
    "geodesic_cost_distance",
    "direct_distance",
    "geodesic_excess",
    "geodesic_ratio",
    "turn_count",
    "turn_rate",
    "first_direction_sin",
    "first_direction_cos",
    "last_direction_sin",
    "last_direction_cos",
    "bottleneck_clearance",
    "mean_clearance",
    "mean_path_cost",
    "path_wall_fraction",
    "path_length",
    "route_hit_fraction",
    "center_route_hit_fraction",
    "route_occupancy_fraction",
    "center_route_occupancy_fraction",
    "route_hit_count_mean",
    "route_hit_count_max",
    "route_hit_count_std",
    "panorama_mean_clearance",
    "panorama_min_clearance",
    "panorama_clearance_std",
    "panorama_blocked_fraction",
    "opposite_clearance_mean",
    "opposite_clearance_max",
)


def _channel_path(paths: RuntimePaths, name: str) -> Path:
    return paths.base_dir / f"{name}.npy"


def _array_path(paths: RuntimePaths, group: str, name: str) -> Path:
    return paths.base_dir / group / f"{name}.npy"


def _complete_memmap(partial: Path, final: Path) -> None:
    if not partial.is_file():
        raise FileNotFoundError(partial)
    os.replace(partial, final)


def _valid_channel(path: Path, count: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, mmap_mode="r")
        return value.shape == (count, 256, 4, 192) and value.dtype == np.complex64
    except (OSError, ValueError):
        return False


def _point_power(channel: np.ndarray, batch_size: int = 16) -> np.ndarray:
    output = np.empty(len(channel), dtype=np.float64)
    for start in range(0, len(channel), batch_size):
        end = min(start + batch_size, len(channel))
        block = np.asarray(channel[start:end], dtype=np.complex64)
        output[start:end] = np.sum(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64)
    return output


def _foundation_configs() -> dict[int, BaselineConfig]:
    return {
        0: BaselineConfig(
            name="affine_k24_steer_axis90_v8pi",
            method="radial_affine",
            k=24,
            ridge=0.30,
            steering_alpha=1.0,
            horizontal_axis_deg=90.0,
            vertical_scale=8.0 * np.pi,
        ),
        1: BaselineConfig(name="radial_idw_k8", method="radial_idw", k=8),
    }


def build_foundation_field(
    paths: RuntimePaths,
    data: OfficialData,
    force: bool = False,
) -> Path:
    """Rebuild the frozen complex interpolation field from official inputs."""

    output = _channel_path(paths, "foundation_field")
    if _valid_channel(output, data.setup.test_count) and not force:
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.npy")
    progress_path = output.with_suffix(".progress.json")
    if force:
        partial.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

    progress = {"completed": {"0": 0, "1": 0}}
    if partial.is_file() and progress_path.is_file():
        try:
            candidate = np.load(partial, mmap_mode="r")
            loaded = json.loads(progress_path.read_text(encoding="utf-8"))
            if (
                candidate.shape == (data.setup.test_count, 256, 4, 192)
                and candidate.dtype == np.complex64
            ):
                progress = loaded
            else:
                partial.unlink(missing_ok=True)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            partial.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)

    if partial.is_file():
        result = np.lib.format.open_memmap(partial, mode="r+")
    else:
        result = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype=np.complex64,
            shape=(data.setup.test_count, 256, 4, 192),
        )
        atomic_json(progress_path, progress)

    train_channel = data.train_channel(mmap_mode="r")
    power_path = _array_path(paths, "foundation", "train_point_power")
    if power_path.is_file():
        point_power = np.load(power_path, allow_pickle=False).astype(
            np.float64, copy=False
        )
        if point_power.shape != (len(train_channel),):
            raise ValueError(f"invalid foundation power cache: {point_power.shape}")
    else:
        point_power = _point_power(train_channel)
        atomic_npy(power_path, point_power)
    all_indices = np.arange(len(data.train_positions), dtype=np.int64)
    carrier = estimate_carrier(
        train_channel,
        data.train_positions,
        data.train_sites,
        data.setup.bs_positions,
        all_indices,
        point_power,
        max_pairs_per_site=500,
        seed=20_260_805,
    )
    templates = _foundation_configs()
    diagnostics: dict[str, Any] = {"sites": {}}

    for site in (0, 1):
        fit_indices = all_indices[data.train_sites == site]
        query_rows = np.flatnonzero(data.test_sites == site)
        completed = int(progress.get("completed", {}).get(str(site), 0))
        if not 0 <= completed <= len(query_rows):
            raise ValueError(
                f"invalid foundation progress for site {site}: {completed}"
            )
        config = replace(
            templates[site], beta=carrier[site].beta, gamma=carrier[site].gamma
        )
        tree = cKDTree(data.train_positions[fit_indices, :2])
        distances, local = tree.query(
            data.test_positions[query_rows, :2], k=config.k, workers=-1
        )
        if config.k == 1:
            distances = distances[:, None]
            local = local[:, None]
        neighbors = fit_indices[local]
        zero_probability: list[float] = []
        zero_count = 0
        for local_row in range(completed, len(query_rows)):
            output_row = int(query_rows[local_row])
            neighbor_index = neighbors[local_row]
            prediction, diagnostic = predict_one(
                np.asarray(train_channel[neighbor_index], dtype=np.complex64),
                data.train_positions[neighbor_index],
                data.test_positions[output_row],
                distances[local_row],
                data.setup.bs_positions[site],
                config,
            )
            result[output_row] = prediction
            zero_probability.append(float(diagnostic["zero_probability"]))
            zero_count += int(bool(diagnostic["zero_gate"]))
            if (local_row + 1) % 16 == 0 or local_row + 1 == len(query_rows):
                result.flush()
                progress["completed"][str(site)] = local_row + 1
                atomic_json(progress_path, progress)
        diagnostics["sites"][str(site)] = {
            "config": config.to_dict(),
            "carrier": carrier[site].to_dict(),
            "zero_gate_count_since_resume": zero_count,
            "zero_probability_mean_since_resume": (
                float(np.mean(zero_probability)) if zero_probability else None
            ),
        }

    result.flush()
    del result
    _complete_memmap(partial, output)
    progress_path.unlink(missing_ok=True)
    atomic_json(paths.report_dir / "foundation_field.json", diagnostics)
    return output


def build_official_spectra(
    paths: RuntimePaths,
    data: OfficialData,
    device: torch.device,
    batch_size: int,
    force: bool = False,
) -> dict[str, Path]:
    root = paths.base_dir / "official_spectra"
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "ad": root / "angle_delay.npy",
        "pas": root / "pas.npy",
        "pdp": root / "pdp.npy",
        "descriptors": root / "descriptors.npz",
        "grouped_pas": root / "grouped_pas.npy",
        "grouped_pdp": root / "grouped_pdp.npy",
    }
    if not force and all(path.is_file() for path in outputs.values()):
        return outputs
    channel = data.train_channel(mmap_mode="r")
    count = len(channel)
    partials = {
        "ad": outputs["ad"].with_suffix(".partial.npy"),
        "pas": outputs["pas"].with_suffix(".partial.npy"),
        "pdp": outputs["pdp"].with_suffix(".partial.npy"),
    }
    for path in partials.values():
        path.unlink(missing_ok=True)
    ad_out = np.lib.format.open_memmap(
        partials["ad"], mode="w+", dtype=np.float16, shape=(count, 16, 8, 192)
    )
    pas_out = np.lib.format.open_memmap(
        partials["pas"], mode="w+", dtype=np.float16, shape=(count, 256, 4, 192)
    )
    pdp_out = np.lib.format.open_memmap(
        partials["pdp"], mode="w+", dtype=np.float16, shape=(count, 256, 4, 192)
    )
    global_pas = np.zeros((count, 256), dtype=np.float32)
    global_pdp = np.zeros((count, 192), dtype=np.float32)
    power = np.zeros(count, dtype=np.float64)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        block = torch.as_tensor(
            np.array(channel[start:end], dtype=np.complex64, copy=True), device=device
        )
        shaped = block.reshape(end - start, 2, 16, 8, 4, 192)
        spatial = torch.fft.fft2(shaped, dim=(2, 3))
        pas_power = spatial.abs().square().float()
        pas_total = pas_power.sum(dim=(1, 2, 3), keepdim=True)
        pas = torch.where(
            pas_total > OFFICIAL_SPECTRAL_TINY,
            pas_power / pas_total.clamp_min(OFFICIAL_SPECTRAL_TINY),
            torch.zeros_like(pas_power),
        )
        delay = torch.fft.ifft(shaped, dim=-1)
        pdp_power = delay.abs().square().float()
        pdp_total = pdp_power.sum(dim=5, keepdim=True)
        pdp = torch.where(
            pdp_total > OFFICIAL_SPECTRAL_TINY,
            pdp_power / pdp_total.clamp_min(OFFICIAL_SPECTRAL_TINY),
            torch.zeros_like(pdp_power),
        )
        angle_delay = torch.fft.ifft(spatial, dim=-1)
        ad_power = angle_delay.abs().square().sum(dim=(1, 4)).float()
        ad_total = ad_power.sum(dim=(1, 2, 3), keepdim=True)
        ad = torch.where(
            ad_total > OFFICIAL_SPECTRAL_TINY,
            ad_power / ad_total.clamp_min(OFFICIAL_SPECTRAL_TINY),
            torch.zeros_like(ad_power),
        )
        pas_numpy = pas.reshape(end - start, 256, 4, 192).cpu().numpy()
        pdp_numpy = pdp.reshape(end - start, 256, 4, 192).cpu().numpy()
        ad_numpy = ad.cpu().numpy()
        ad_out[start:end] = ad_numpy.astype(np.float16)
        pas_out[start:end] = pas_numpy.astype(np.float16)
        pdp_out[start:end] = pdp_numpy.astype(np.float16)
        global_pas[start:end] = pas_numpy.mean(axis=(2, 3), dtype=np.float64)
        global_pdp[start:end] = pdp_numpy.mean(axis=(1, 2), dtype=np.float64)
        power[start:end] = (
            block.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64).cpu().numpy()
        )
    for value in (ad_out, pas_out, pdp_out):
        value.flush()
    del ad_out, pas_out, pdp_out
    for name in ("ad", "pas", "pdp"):
        _complete_memmap(partials[name], outputs[name])
    valid = power > TINY
    atomic_npz(
        outputs["descriptors"],
        global_pas=global_pas,
        global_pdp=global_pdp,
        total_power=power,
        valid=valid,
        pas_entropy=-np.sum(
            global_pas * np.log(np.maximum(global_pas, TINY)), axis=1
        ).astype(np.float32),
        pdp_entropy=-np.sum(
            global_pdp * np.log(np.maximum(global_pdp, TINY)), axis=1
        ).astype(np.float32),
    )
    grouped_pas_partial = outputs["grouped_pas"].with_suffix(".partial.npy")
    grouped_pdp_partial = outputs["grouped_pdp"].with_suffix(".partial.npy")
    grouped_pas_partial.unlink(missing_ok=True)
    grouped_pdp_partial.unlink(missing_ok=True)
    grouped_pas = np.lib.format.open_memmap(
        grouped_pas_partial, mode="w+", dtype=np.float16, shape=(count, 32, 256)
    )
    grouped_pdp = np.lib.format.open_memmap(
        grouped_pdp_partial, mode="w+", dtype=np.float16, shape=(count, 8, 192)
    )
    pas_source = np.load(outputs["pas"], mmap_mode="r")
    pdp_source = np.load(outputs["pdp"], mmap_mode="r")
    for start in range(0, count, 16):
        end = min(start + 16, count)
        pas = np.asarray(pas_source[start:end], dtype=np.float32)
        pas = (
            pas.reshape(end - start, 256, 4, 8, 24)
            .mean(axis=4, dtype=np.float64)
            .astype(np.float32)
        )
        pas = np.transpose(pas, (0, 2, 3, 1)).reshape(end - start, 32, 256)
        pdp = np.asarray(pdp_source[start:end], dtype=np.float32)
        pdp = (
            pdp.reshape(end - start, 2, 16, 8, 4, 192)
            .mean(axis=(2, 3), dtype=np.float64)
            .astype(np.float32)
            .reshape(end - start, 8, 192)
        )
        grouped_pas[start:end] = np.sqrt(np.maximum(pas, 0.0)).astype(np.float16)
        grouped_pdp[start:end] = np.sqrt(np.maximum(pdp, 0.0)).astype(np.float16)
    grouped_pas.flush()
    grouped_pdp.flush()
    del grouped_pas, grouped_pdp
    _complete_memmap(grouped_pas_partial, outputs["grouped_pas"])
    _complete_memmap(grouped_pdp_partial, outputs["grouped_pdp"])
    return outputs


def build_carrier_bank(
    paths: RuntimePaths,
    data: OfficialData,
    config: PipelineConfig,
    device: torch.device,
    force: bool = False,
) -> Path:
    output = _array_path(paths, "carriers", "bank")
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = (17, data.setup.test_count, 256, 4, 192)
    if output.is_file() and not force:
        cached = np.load(output, mmap_mode="r")
        if cached.shape == expected and cached.dtype == np.complex64:
            return output
    partial = output.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    train_channel = data.train_channel(mmap_mode="r")
    device_channel = (
        torch.as_tensor(np.asarray(train_channel), dtype=torch.complex64, device=device)
        if device.type == "cuda"
        else None
    )
    previous = carrier_bank.TEST_BANK
    carrier_bank.TEST_BANK = partial
    try:
        value, _ = carrier_bank.build_test_bank(
            train_channel,
            data.train_positions,
            data.test_positions,
            data.train_sites,
            data.test_sites,
            data.setup.bs_positions,
            _point_power(train_channel),
            carrier_bank.carrier_configs(),
            carrier_pairs=500,
            workers=1,
            device_channel=device_channel,
            gpu_batch_size=config.inference_batch_size,
        )
        if value.shape != expected or value.dtype != np.complex64:
            raise ValueError(f"invalid carrier bank: {value.shape}/{value.dtype}")
        del value
    finally:
        carrier_bank.TEST_BANK = previous
        del device_channel
    _complete_memmap(partial, output)
    return output


def combine_carriers(
    paths: RuntimePaths,
    data: OfficialData,
    bank_path: Path,
    batch_size: int,
    force: bool = False,
) -> Path:
    output = _channel_path(paths, "carrier_mixture")
    if _valid_channel(output, data.setup.test_count) and not force:
        return output
    with np.load(
        paths.model_dir / "base_field" / "carrier_mix_weights.npz",
        allow_pickle=False,
    ) as loaded:
        weights = {
            int(site): np.asarray(loaded[f"label_{site}"], dtype=np.complex64)
            for site in (0, 1)
        }
    bank = np.load(bank_path, mmap_mode="r")
    partial = output.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    result = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.complex64,
        shape=(data.setup.test_count, 256, 4, 192),
    )
    for start in range(0, len(result), batch_size):
        end = min(start + batch_size, len(result))
        rows = np.arange(start, end)
        block = np.asarray(bank[:, rows], dtype=np.complex64)
        candidates = (
            block.reshape(17, len(rows), 2, 16, 8, 4, 192)
            .transpose(1, 2, 5, 0, 3, 4, 6)
            .reshape(len(rows), 8, 17, 16, 8, 192)
        )
        local_weights = np.stack(
            [weights[int(data.test_sites[row])] for row in rows], axis=0
        )
        mixed = np.einsum("bgcvhd,bgc->bgvhd", candidates, local_weights, optimize=True)
        result[start:end] = (
            mixed.reshape(len(rows), 2, 4, 16, 8, 192)
            .transpose(0, 1, 3, 4, 2, 5)
            .reshape(len(rows), 256, 4, 192)
            .astype(np.complex64)
        )
    result.flush()
    del result
    _complete_memmap(partial, output)
    return output


def _group_calibration(
    paths: RuntimePaths,
    data: OfficialData,
    spectra: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray]:
    deployment = np.load(
        paths.model_dir / "base_field" / "deployment_donor_weights.npz",
        allow_pickle=False,
    )
    donor_ids = deployment["donor_ids"].astype(np.int64)
    pas_weights = deployment["pas_weights"].astype(np.float32)
    pdp_weights = deployment["pdp_weights"].astype(np.float32)
    grouped_pas = predict_grouped(
        np.load(spectra["grouped_pas"], mmap_mode="r"), donor_ids, pas_weights
    )
    grouped_pdp = predict_grouped(
        np.load(spectra["grouped_pdp"], mmap_mode="r"), donor_ids, pdp_weights
    )
    with np.load(
        paths.model_dir / "base_field" / "group_calibration.npz",
        allow_pickle=False,
    ) as loaded:
        pas_models = {
            site: (
                loaded[f"site{site}_pas_mean"],
                loaded[f"site{site}_pas_vectors"],
                loaded[f"site{site}_pas_slope"],
            )
            for site in (0, 1)
        }
        pdp_models = {
            site: (
                loaded[f"site{site}_pdp_mean"],
                loaded[f"site{site}_pdp_vectors"],
                loaded[f"site{site}_pdp_slope"],
            )
            for site in (0, 1)
        }
    return (
        apply_calibration(grouped_pas, data.test_sites, pas_models, 1.0),
        apply_calibration(grouped_pdp, data.test_sites, pdp_models, 0.5),
    )


def project_official_targets(
    paths: RuntimePaths,
    data: OfficialData,
    spectra: dict[str, Path],
    base_path: Path,
    device: torch.device,
    batch_size: int,
    output_name: str,
    policy_key: str,
    fallback_scales: dict[int, float],
    force: bool = False,
) -> Path:
    output = _channel_path(paths, output_name)
    if _valid_channel(output, data.setup.test_count) and not force:
        return output
    deployment = np.load(
        paths.model_dir / "base_field" / "deployment_donor_weights.npz",
        allow_pickle=False,
    )
    donor_ids = deployment["donor_ids"].astype(np.int64)
    pas_weights = deployment["pas_weights"].astype(np.float32)
    pdp_weights = deployment["pdp_weights"].astype(np.float32)
    calibrated_pas, calibrated_pdp = _group_calibration(paths, data, spectra)
    train_ad = np.load(spectra["ad"], mmap_mode="r")
    train_pas = np.load(spectra["pas"], mmap_mode="r")
    train_pdp = np.load(spectra["pdp"], mmap_mode="r")
    base = np.load(base_path, mmap_mode="r")
    policy_payload = json.loads(
        (paths.model_dir / "base_field" / "projection_policy.json").read_text(
            encoding="utf-8"
        )
    )
    has_split_policies = "reference_policy" in policy_payload
    policy_rows = policy_payload.get(policy_key, policy_payload["final_policy"])
    policy = {int(row["site"]): str(row["candidate"]) for row in policy_rows}
    site_scales = {
        int(row["site"]): (
            float(row.get("scale", fallback_scales[int(row["site"])]))
            if has_split_policies
            else float(fallback_scales[int(row["site"])])
        )
        for row in policy_rows
    }
    partial = output.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    result = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        ad_rows = []
        pas_rows = []
        pdp_rows = []
        for row in range(start, end):
            ad_rows.append(
                spectral_target(train_ad, donor_ids[row], pas_weights[row], (0, 1, 2))
            )
            pas_rows.append(
                spectral_target(train_pas, donor_ids[row], pas_weights[row], 0)
            )
            pdp_rows.append(
                spectral_target(train_pdp, donor_ids[row], pdp_weights[row], 2)
            )
        raw_pas, raw_pdp = globally_shrunk_targets(
            np.stack(pas_rows), np.stack(pdp_rows)
        )
        local_pas = blend_group_pas(raw_pas, calibrated_pas[start:end], 0.25)
        local_pdp = blend_group_pdp(raw_pdp, calibrated_pdp[start:end], 1.0)
        ad_tensor = torch.as_tensor(np.stack(ad_rows), device=device)
        raw_pas_tensor = torch.as_tensor(raw_pas, device=device)
        raw_pdp_tensor = torch.as_tensor(raw_pdp, device=device)
        calibrated_pas_tensor = torch.as_tensor(local_pas, device=device)
        calibrated_pdp_tensor = torch.as_tensor(local_pdp, device=device)
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        block = np.empty((end - start, 256, 4, 192), dtype=np.complex64)
        local_sites = data.test_sites[start:end]
        with torch.inference_mode():
            for site in np.unique(local_sites):
                rows = np.flatnonzero(local_sites == site)
                match = EXACT_PATTERN.fullmatch(policy[int(site)])
                if match is None:
                    raise ValueError(
                        f"unsupported projection policy: {policy[int(site)]}"
                    )
                mode, order, action_name, cycles_text = match.groups()
                action_index = next(
                    index
                    for index, action in enumerate(ACTIONS)
                    if action.name == action_name
                )
                desired_pas, desired_pdp = mode_targets(
                    mode,
                    raw_pas_tensor[rows],
                    raw_pdp_tensor[rows],
                    calibrated_pas_tensor[rows],
                    calibrated_pdp_tensor[rows],
                )
                prediction, target_pas, target_pdp, reference_power = (
                    initialize_trajectory(
                        base_tensor[rows],
                        ad_tensor[rows],
                        desired_pas,
                        desired_pdp,
                        action_index,
                        order,
                    )
                )
                for _ in range(2, int(cycles_text) + 1):
                    prediction = continue_trajectory(
                        prediction, target_pas, target_pdp, reference_power, order
                    )
                block[rows] = (
                    prediction.cpu().numpy() * np.float32(site_scales[int(site)])
                ).astype(np.complex64)
        result[start:end] = block
    result.flush()
    del result
    _complete_memmap(partial, output)
    return output


def build_lifecycle_field(
    paths: RuntimePaths,
    data: OfficialData,
    spectra: dict[str, Path],
    projected_path: Path,
    reference_projected_path: Path,
    device: torch.device,
    batch_size: int,
    force: bool = False,
) -> Path:
    output = _channel_path(paths, "lifecycle_field")
    if _valid_channel(output, data.setup.test_count) and not force:
        return output
    geometry = build_geometry_regime(paths, force=False)
    policy_payload = json.loads(
        (paths.model_dir / "base_field" / "lifecycle_policy.json").read_text(
            encoding="utf-8"
        )
    )
    winner = policy_payload.get("winner") or {}
    policy_rows = winner.get("final_policy", [])
    policy = {
        (int(row["site"]), int(row["regime_index"])): str(row["candidate"])
        for row in policy_rows
    }
    if not policy or all(name == "identity" for name in policy.values()):
        base = np.load(projected_path, mmap_mode="r")
        partial = output.with_suffix(".partial.npy")
        partial.unlink(missing_ok=True)
        result = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.complex64, shape=base.shape
        )
        for start in range(0, len(base), batch_size):
            end = min(start + batch_size, len(base))
            result[start:end] = np.asarray(base[start:end], dtype=np.complex64)
        result.flush()
        del result
        _complete_memmap(partial, output)
        return output

    candidate_by_name = {
        candidate.name: candidate for candidate in lifecycle.candidate_list()
    }
    required_targets = {
        candidate_by_name[name].target
        for name in policy.values()
        if name != "identity"
    }
    current_path = _array_path(paths, "lifecycle", "current_angle_delay")
    current_path.parent.mkdir(parents=True, exist_ok=True)
    if force:
        current_path.unlink(missing_ok=True)
    current = lifecycle.channel_ad_support(
        np.load(projected_path, mmap_mode="r"), current_path, device, batch_size
    )
    target_paths = {
        name: _array_path(paths, "lifecycle", f"target_{name}")
        for name in lifecycle.TARGET_PATHS
    }
    if force or not all(target_paths[name].is_file() for name in required_targets):
        reference_current_path = _array_path(
            paths, "lifecycle", "reference_angle_delay"
        )
        if force:
            reference_current_path.unlink(missing_ok=True)
        reference_current = lifecycle.channel_ad_support(
            np.load(reference_projected_path, mmap_mode="r"),
            reference_current_path,
            device,
            batch_size,
        )
        truth_support = lifecycle.normalize_support(
            np.load(spectra["ad"], mmap_mode="r")
        )
        current_value = lifecycle.normalize_support(reference_current)
        train_features = np.asarray(geometry["train_features"], dtype=np.float32)
        test_features = np.asarray(geometry["test_features"], dtype=np.float32)
        model_train = np.concatenate(
            (
                np.asarray(geometry["train_visibility"], dtype=np.float32),
                train_features,
            ),
            axis=1,
        )
        model_test = np.concatenate(
            (
                np.asarray(geometry["test_visibility"], dtype=np.float32),
                test_features,
            ),
            axis=1,
        )
        full_fit = np.arange(len(data.train_positions), dtype=np.int64)
        train_risk, test_risk, train_regime, test_regime, _ = (
            lifecycle.physical_risk_and_regime(
                train_features,
                test_features,
                data.train_sites,
                data.test_sites,
                full_fit,
                GEODESIC_NAMES,
            )
        )
        truth_descriptor = lifecycle.support_descriptor(truth_support)
        current_descriptor = lifecycle.support_descriptor(current_value)
        template, _, _ = lifecycle.build_regime_template(
            truth_support,
            truth_descriptor,
            current_value,
            current_descriptor,
            model_train,
            model_test,
            data.train_sites,
            data.test_sites,
            train_regime,
            test_regime,
            full_fit,
            20_260_906,
        )
        train_route = (
            np.asarray(geometry["train_route_occupancy"], dtype=np.float32).reshape(
                len(data.train_positions), -1
            ),
            np.asarray(geometry["train_panorama"], dtype=np.float32).reshape(
                len(data.train_positions), -1
            ),
        )
        test_route = (
            np.asarray(geometry["test_route_occupancy"], dtype=np.float32).reshape(
                len(data.test_positions), -1
            ),
            np.asarray(geometry["test_panorama"], dtype=np.float32).reshape(
                len(data.test_positions), -1
            ),
        )
        plain, visibility, shifted, _ = lifecycle.build_lifecycle_targets(
            current_value,
            template,
            data.test_positions,
            data.test_sites,
            np.arange(len(data.test_positions), dtype=np.int64),
            full_fit,
            data.train_positions,
            data.train_sites,
            lifecycle.extract_peaks(truth_support),
            lifecycle.extract_peaks(current_value),
            lifecycle.extract_peaks(template),
            train_regime,
            test_regime,
            train_risk,
            test_risk,
            train_features,
            test_features,
            train_route,
            test_route,
        )
        targets = {
            "template": template,
            "lifecycle_plain": plain,
            "lifecycle_visibility": visibility,
            "lifecycle_shift": shifted,
        }
        for name, value in targets.items():
            target_path = target_paths[name]
            partial_target = target_path.with_suffix(".partial.npy")
            partial_target.unlink(missing_ok=True)
            target_output = np.lib.format.open_memmap(
                partial_target,
                mode="w+",
                dtype=np.float16,
                shape=value.shape,
            )
            target_output[:] = lifecycle.normalize_support(value).astype(np.float16)
            target_output.flush()
            del target_output
            _complete_memmap(partial_target, target_path)
    target_arrays = {
        name: np.load(target_paths[name], mmap_mode="r") for name in required_targets
    }
    regimes = np.asarray(geometry["test_regime"], dtype=np.int8)
    base = np.load(projected_path, mmap_mode="r")
    partial = output.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    result = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        block = np.array(base[start:end], dtype=np.complex64, copy=True)
        local_sites = data.test_sites[start:end]
        local_regimes = regimes[start:end]
        for site in np.unique(local_sites):
            for regime in np.unique(local_regimes):
                rows = np.flatnonzero((local_sites == site) & (local_regimes == regime))
                if not len(rows):
                    continue
                name = policy.get((int(site), int(regime)), "identity")
                if name == "identity":
                    continue
                candidate = candidate_by_name[name]
                with torch.inference_mode():
                    prediction = lifecycle.apply_candidate(
                        torch.as_tensor(block[rows], device=device),
                        torch.as_tensor(
                            np.asarray(current[start:end][rows], dtype=np.float32),
                            device=device,
                        ),
                        torch.as_tensor(
                            np.asarray(
                                target_arrays[candidate.target][start:end][rows],
                                dtype=np.float32,
                            ),
                            device=device,
                        ),
                        local_regimes[rows],
                        local_sites[rows],
                        candidate,
                    )
                block[rows] = prediction.cpu().numpy().astype(np.complex64)
        result[start:end] = block
    result.flush()
    del result
    _complete_memmap(partial, output)
    return output


def _physical_stat_path(
    paths: RuntimePaths,
    configuration: str,
    split: str,
    domain: str,
    statistic: str,
) -> Path:
    return (
        paths.work_dir
        / "physical_spectra"
        / split
        / configuration
        / f"{domain}_{statistic}.npy"
    )


def _legacy_position_tuple(data: OfficialData) -> tuple[Any, ...]:
    return (
        data.setup,
        data.train_positions,
        data.test_positions,
        data.train_sites,
        data.test_sites,
    )


def _visibility_cache(paths: RuntimePaths, geometry: dict[str, np.ndarray]) -> Path:
    output = paths.base_dir / "visibility_features.npz"
    if not output.is_file():
        atomic_npz(
            output,
            train=np.asarray(geometry["train_visibility"], dtype=np.float32),
            test=np.asarray(geometry["test_visibility"], dtype=np.float32),
        )
    return output


def _configure_spectral_modules(
    paths: RuntimePaths,
    data: OfficialData,
    config: PipelineConfig,
    geometry: dict[str, np.ndarray],
) -> None:
    visibility = _visibility_cache(paths, geometry)

    def running_path(
        configuration: str, split: str, domain: str, statistic: str
    ) -> Path:
        name = "scatter" if "scatter" in str(configuration).lower() else "base"
        return _physical_stat_path(paths, name, split, domain, statistic)

    load_positions = lambda: _legacy_position_tuple(data)
    spectral_stage.GEOMETRY_CACHE = visibility
    spectral_stage.BASE_CONFIG = "base"
    spectral_stage.SEEDS = len(config.seeds)
    spectral_stage.running_path = running_path
    spectral_stage.load_positions = load_positions
    spectral_stage.model_path = lambda domain, mode, fold: (
        paths.model_dir / "base_field" / "spectral" / f"{domain}_fold_{fold}.pt"
    )

    joint_stage.GEOMETRY_CACHE = visibility
    joint_stage.BASE_CONFIG = "base"
    joint_stage.SCATTER_CONFIG = "scatter"
    joint_stage.SEEDS = len(config.seeds)
    joint_stage.running_path = running_path
    joint_stage.load_positions = load_positions
    joint_stage.ARTIFACT_DIR = paths.base_dir / "joint_endpoint"
    joint_stage.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joint_stage.reliability_path = lambda domain, split: (
        paths.base_dir / "joint_endpoint" / f"{split}_{domain}_reliability.npy"
    )
    joint_stage.checkpoint_path = lambda domain, fold: (
        paths.model_dir / "base_field" / "joint" / f"{domain}_fold_{fold}.pt"
    )
    joint_stage.output_path = lambda domain, split: (
        paths.base_dir / "joint_endpoint" / f"{split}_{domain}.npy"
    )


def build_spectral_endpoints(
    paths: RuntimePaths,
    data: OfficialData,
    config: PipelineConfig,
    device: torch.device,
    inference_batch_size: int,
    force: bool = False,
) -> dict[str, Path]:
    output_dir = paths.base_dir / "spectral_endpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "calibrated_pas": output_dir / "calibrated_pas.npy",
        "calibrated_pdp": output_dir / "calibrated_pdp.npy",
        "joint_pas": output_dir / "joint_pas.npy",
        "joint_pdp": output_dir / "joint_pdp.npy",
        "pas_reliability": paths.base_dir
        / "joint_endpoint"
        / "test_pas_reliability.npy",
        "pdp_reliability": paths.base_dir
        / "joint_endpoint"
        / "test_pdp_reliability.npy",
    }
    if not force and all(path.is_file() for path in outputs.values()):
        return outputs
    geometry = build_geometry_regime(paths, force=False)
    _configure_spectral_modules(paths, data, config, geometry)
    for domain in ("pas", "pdp"):
        target_path = outputs[f"calibrated_{domain}"]
        if target_path.is_file() and not force:
            continue
        value, _ = spectral_stage.ensemble_target(domain, device, inference_batch_size)
        if domain == "pas":
            shaped = value.reshape(500, 4, 192, 2, 16, 8).transpose(0, 3, 4, 5, 1, 2)
        else:
            shaped = value.reshape(500, 256, 4, 192)
        temporary = target_path.with_suffix(".tmp.npy")
        np.save(temporary, shaped.astype(np.float16))
        os.replace(temporary, target_path)
        del value, shaped
        if device.type == "cuda":
            torch.cuda.empty_cache()
    generated = joint_stage.generate_test_targets(
        ("pas", "pdp"), device, inference_batch_size, force
    )
    del generated
    for domain in ("pas", "pdp"):
        source = joint_stage.output_path(domain, "test")
        destination = outputs[f"joint_{domain}"]
        if source != destination:
            os.replace(source, destination)
    for domain in ("pas", "pdp"):
        reliability = joint_stage.spectral_reliability(domain, "test", force=False)
        expected = outputs[f"{domain}_reliability"]
        if Path(reliability.filename) != expected:
            np.save(expected, np.asarray(reliability, dtype=np.float32))
    return outputs


def build_joint_spectral_field(
    paths: RuntimePaths,
    data: OfficialData,
    lifecycle_path: Path,
    endpoints: dict[str, Path],
    device: torch.device,
    batch_size: int,
    force: bool = False,
) -> Path:
    output = _channel_path(paths, "joint_spectral_field")
    if _valid_channel(output, data.setup.test_count) and not force:
        return output
    base = np.load(lifecycle_path, mmap_mode="r")
    old_pas = np.load(endpoints["calibrated_pas"], mmap_mode="r")
    old_pdp = np.load(endpoints["calibrated_pdp"], mmap_mode="r")
    joint_pas = np.load(endpoints["joint_pas"], mmap_mode="r")
    joint_pdp = np.load(endpoints["joint_pdp"], mmap_mode="r")
    policy = JointCandidate(
        name="frozen_joint_endpoint",
        pas_beta=0.75,
        pdp_beta=0.50,
        alpha_pas=0.45,
        alpha_pdp=0.35,
        order="pas_pdp",
        cycles=4,
    )
    partial = output.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    result = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        with torch.inference_mode():
            prediction = apply_joint_candidate(
                torch.as_tensor(
                    np.array(base[start:end], dtype=np.complex64, copy=True),
                    device=device,
                ),
                torch.as_tensor(
                    np.asarray(old_pas[start:end], dtype=np.float32), device=device
                ),
                torch.as_tensor(
                    np.asarray(joint_pas[start:end], dtype=np.float32), device=device
                ),
                torch.as_tensor(
                    np.asarray(old_pdp[start:end], dtype=np.float32), device=device
                ),
                torch.as_tensor(
                    np.asarray(joint_pdp[start:end], dtype=np.float32), device=device
                ),
                policy,
            )
        result[start:end] = prediction.cpu().numpy().astype(np.complex64)
    result.flush()
    del result
    _complete_memmap(partial, output)
    return output


def build_path_link_field(
    paths: RuntimePaths,
    data: OfficialData,
    joint_field_path: Path,
    endpoints: dict[str, Path],
    device: torch.device,
    batch_size: int,
    force: bool = False,
) -> Path:
    output = _channel_path(paths, "path_link_field")
    if _valid_channel(output, data.setup.test_count) and not force:
        return output
    features, _ = load_model_inputs(paths, "test")
    geometry = build_geometry_regime(paths, force=False)
    stage_dir = paths.base_dir / "path_link"
    stage_dir.mkdir(parents=True, exist_ok=True)
    regime_path = stage_dir / "regime.npy"
    if not regime_path.is_file() or force:
        np.save(regime_path, np.asarray(geometry["test_regime"], dtype=np.int8))
    path_link_stage.PATH_CACHE = path_token_dir(paths, "test")
    path_link_stage.FEATURES = Path(features.filename)
    path_link_stage.REGIME = regime_path
    path_link_stage.RELIABILITY = endpoints["pdp_reliability"]
    path_link_stage.JOINT_TEST = joint_field_path
    path_link_stage.ARTIFACT_DIR = stage_dir
    path_link_stage.BASE_PDP = stage_dir / "base_pdp.npy"
    path_link_stage.RAW_PDP = stage_dir / "raw_path_pdp.npy"
    path_link_stage.POINT_CONTEXT = stage_dir / "point_context.npy"
    path_link_stage.FOLD_ALPHA = stage_dir / "fold_alpha.npy"
    path_link_stage.ENSEMBLE_ALPHA = stage_dir / "ensemble_alpha.npy"
    path_link_stage.OUTPUT = output
    path_link_stage.load_positions = lambda: _legacy_position_tuple(data)
    path_link_model.checkpoint_path = lambda fold: (
        paths.model_dir / "base_field" / "path_link" / f"fold_{fold}.pt"
    )
    if force:
        for path in (
            path_link_stage.BASE_PDP,
            path_link_stage.RAW_PDP,
            path_link_stage.POINT_CONTEXT,
            path_link_stage.FOLD_ALPHA,
            path_link_stage.ENSEMBLE_ALPHA,
            output,
        ):
            path.unlink(missing_ok=True)
    base_pdp = path_link_stage.build_base_pdp(device, batch_size, force=False)
    raw_pdp = path_link_stage.build_raw_path_pdp(device, batch_size, force=False)
    context = path_link_stage.build_point_context(force=False)
    alpha, _ = path_link_stage.predict_ensemble_alpha(
        base_pdp, raw_pdp, context, device, batch_size
    )
    path_link_stage.generate(base_pdp, raw_pdp, alpha, device, batch_size)
    return output


def build_reliable_base_field(
    paths: RuntimePaths,
    data: OfficialData,
    path_link_field: Path,
    endpoints: dict[str, Path],
    device: torch.device,
    batch_size: int,
    force: bool = False,
) -> Path:
    output = _channel_path(paths, "base_channel")
    if _valid_channel(output, data.setup.test_count) and not force:
        return output
    stage_dir = paths.base_dir / "reliability"
    stage_dir.mkdir(parents=True, exist_ok=True)
    reliability_stage.PATH_LINK_TEST = path_link_field
    reliability_stage.CALIBRATED_PAS = endpoints["calibrated_pas"]
    reliability_stage.JOINT_ENDPOINT_TEST = endpoints["joint_pas"]
    reliability_stage.BASE_PHYSICAL_PAS = _physical_stat_path(
        paths, "base", "test", "pas", "mean"
    )
    reliability_stage.SCATTER_PHYSICAL_PAS = _physical_stat_path(
        paths, "scatter", "test", "pas", "mean"
    )
    reliability_stage.RAW_FEATURES = stage_dir / "raw_features.npy"
    reliability_stage.GATE_SCORE = stage_dir / "gate_score.npy"
    reliability_stage.GATE = stage_dir / "gate.npy"
    reliability_stage.OUTPUT = output
    if force:
        for path in (
            reliability_stage.RAW_FEATURES,
            reliability_stage.GATE_SCORE,
            reliability_stage.GATE,
            output,
        ):
            path.unlink(missing_ok=True)
    base = np.load(path_link_field, mmap_mode="r")
    sources = reliability_stage.source_arrays()
    gate = reliability_stage.build_gate(base, sources, device, batch_size, force=False)
    reliability_stage.generate(base, sources, gate, device, batch_size)
    return output


def build_base_field(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    started = time.time()
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    data = load_official_data(paths.data_dir)
    required_physics = [
        _physical_stat_path(paths, configuration, "test", domain, statistic)
        for configuration in ("base", "scatter")
        for domain in ("pas", "pdp", "ad", "log_power", "path_count")
        for statistic in ("mean", "m2", "count")
    ]
    missing_physics = [str(path) for path in required_physics if not path.is_file()]
    if missing_physics:
        raise FileNotFoundError(
            "physical spectral aggregation must run before the base field; missing "
            f"{len(missing_physics)} files, first={missing_physics[0]}"
        )
    phase_seconds: dict[str, float] = {}

    phase = time.time()
    official = build_official_spectra(paths, data, device, batch_size, force)
    phase_seconds["official_spectra"] = time.time() - phase

    phase = time.time()
    foundation = build_foundation_field(paths, data, force)
    reference_projected = project_official_targets(
        paths,
        data,
        official,
        foundation,
        device,
        batch_size,
        "lifecycle_reference_projection",
        "reference_policy",
        REFERENCE_FIELD_SCALES,
        force,
    )
    phase_seconds["foundation_reference"] = time.time() - phase

    phase = time.time()
    bank = build_carrier_bank(paths, data, config, device, force)
    mixture = combine_carriers(paths, data, bank, batch_size, force)
    phase_seconds["complex_carriers"] = time.time() - phase

    phase = time.time()
    projected = project_official_targets(
        paths,
        data,
        official,
        mixture,
        device,
        batch_size,
        "official_spectral_projection",
        "final_policy",
        STACKED_FIELD_SCALES,
        force,
    )
    lifecycle_field = build_lifecycle_field(
        paths,
        data,
        official,
        projected,
        reference_projected,
        device,
        batch_size,
        force,
    )
    phase_seconds["official_projection_and_lifecycle"] = time.time() - phase

    phase = time.time()
    endpoints = build_spectral_endpoints(
        paths, data, config, device, config.inference_batch_size, force
    )
    joint = build_joint_spectral_field(
        paths, data, lifecycle_field, endpoints, device, batch_size, force
    )
    phase_seconds["physical_spectral_models"] = time.time() - phase

    phase = time.time()
    linked = build_path_link_field(
        paths, data, joint, endpoints, device, batch_size, force
    )
    final = build_reliable_base_field(
        paths, data, linked, endpoints, device, batch_size, force
    )
    phase_seconds["path_link_and_reliability"] = time.time() - phase

    validation = validate_channel(final, data.setup.test_count)
    report = {
        "stage": "base_field",
        "status": "complete",
        "output": validation,
        "historical_complete_channel_read": False,
        "official_train_channel_used": True,
        "local_physical_spectra_used": True,
        "seed_count": config.seed_count,
        "phase_seconds": phase_seconds,
        "wall_seconds": float(time.time() - started),
        "device": str(device),
        "peak_torch_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
        "peak_rss_gib": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        ),
        "measured_bottleneck": max(phase_seconds, key=phase_seconds.get),
    }
    atomic_json(paths.report_dir / "base_field.json", report)
    return report
