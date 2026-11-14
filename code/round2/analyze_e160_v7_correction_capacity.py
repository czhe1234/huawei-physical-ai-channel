#!/usr/bin/env python3
"""Test the correction-space capacity of the strict Round2-E160 V7 library."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
)
from analyze_v7_stacked_lifecycle_overlay import (
    OOF_SCALED,
    SELECTION as LIFECYCLE_SELECTION,
)
from data import load_positions
from official_metrics import OfficialMetricAccumulator
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META
from run_v6_regime_component_lifecycle import (
    REGIME_CACHE,
    REGIME_NAMES,
    TARGET_PATHS,
    _candidate_from_name,
    apply_candidate,
)
from train_full_path_pdp_link_gate import FOLD_IDS


VERSION = "Round2-E160-v7-correction-capacity"
E160_DIR = PATHS.artifacts / "round2" / "e160_full4000_v7_supervision"
ARTIFACT_DIR = E160_DIR / "correction_capacity"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e160_full4000_v7_supervision"
FULL_CORE = E160_DIR / "full4000_v7_core.npy"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

FULL_CORE_PAS = ARTIFACT_DIR / "full4000_core_grouped_pas_f16.npy"
FULL_CORE_PDP = ARTIFACT_DIR / "full4000_core_grouped_pdp_f16.npy"
FULL_TRUTH_PAS = ARTIFACT_DIR / "full4000_truth_grouped_pas_f16.npy"
FULL_TRUTH_PDP = ARTIFACT_DIR / "full4000_truth_grouped_pdp_f16.npy"
FULL_RESIDUAL_PAS = ARTIFACT_DIR / "full4000_root_residual_pas_f16.npy"
FULL_RESIDUAL_PDP = ARTIFACT_DIR / "full4000_root_residual_pdp_f16.npy"
REPORT = ANALYSIS_DIR / "full4000_v7_correction_capacity.json"

PAS_GROUPS = 8
PDP_H_GROUPS = 4
PDP_V_GROUPS = 2
PAS_GROUP_SIZE = 192 // PAS_GROUPS
PDP_H_SIZE = 16 // PDP_H_GROUPS
PDP_V_SIZE = 8 // PDP_V_GROUPS
TINY = 1e-30
REQUIRED_SCORE = 0.69


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--force-endpoints", action="store_true")
    return parser.parse_args()


def normalize(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(
        total > TINY,
        value / total.clamp_min(TINY),
        torch.zeros_like(value),
    )


@torch.inference_mode()
def grouped_endpoints(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(value)
    pas = pas_distribution(value).reshape(batch, 2, 16, 8, 4, 192)
    pas = pas.reshape(batch, 2, 16, 8, 4, PAS_GROUPS, PAS_GROUP_SIZE)
    pas = normalize(pas.mean(dim=6), (1, 2, 3))

    pdp = pdp_distribution(value).reshape(batch, 2, 16, 8, 4, 192)
    pdp = pdp.reshape(
        batch,
        2,
        PDP_H_GROUPS,
        PDP_H_SIZE,
        PDP_V_GROUPS,
        PDP_V_SIZE,
        4,
        192,
    )
    pdp = normalize(pdp.mean(dim=(3, 5)), (5,))
    return pas.float(), pdp.float()


def endpoint_paths(prefix: str) -> tuple[object, object]:
    if prefix == "core":
        return FULL_CORE_PAS, FULL_CORE_PDP
    if prefix == "truth":
        return FULL_TRUTH_PAS, FULL_TRUTH_PDP
    raise ValueError(prefix)


def build_endpoint_cache(
    channel: np.ndarray,
    prefix: str,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    pas_path, pdp_path = endpoint_paths(prefix)
    pas_shape = (len(channel), 2, 16, 8, 4, PAS_GROUPS)
    pdp_shape = (len(channel), 2, PDP_H_GROUPS, PDP_V_GROUPS, 4, 192)
    if pas_path.is_file() and pdp_path.is_file() and not force:
        pas = np.load(pas_path, mmap_mode="r")
        pdp = np.load(pdp_path, mmap_mode="r")
        if pas.shape == pas_shape and pdp.shape == pdp_shape:
            return pas, pdp
    pas_output = np.lib.format.open_memmap(
        pas_path, mode="w+", dtype=np.float16, shape=pas_shape
    )
    pdp_output = np.lib.format.open_memmap(
        pdp_path, mode="w+", dtype=np.float16, shape=pdp_shape
    )
    for start in range(0, len(channel), batch_size):
        end = min(start + batch_size, len(channel))
        tensor = torch.as_tensor(
            np.array(channel[start:end], dtype=np.complex64, copy=True), device=device
        )
        pas, pdp = grouped_endpoints(tensor)
        pas_output[start:end] = pas.cpu().numpy().astype(np.float16)
        pdp_output[start:end] = pdp.cpu().numpy().astype(np.float16)
        if end % (batch_size * 20) == 0 or end == len(channel):
            print(f"E160 {prefix} endpoints {end}/{len(channel)}", flush=True)
    pas_output.flush()
    pdp_output.flush()
    return np.load(pas_path, mmap_mode="r"), np.load(pdp_path, mmap_mode="r")


def build_residual_cache(
    core_pas: np.ndarray,
    core_pdp: np.ndarray,
    truth_pas: np.ndarray,
    truth_pdp: np.ndarray,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if FULL_RESIDUAL_PAS.is_file() and FULL_RESIDUAL_PDP.is_file() and not force:
        return (
            np.load(FULL_RESIDUAL_PAS, mmap_mode="r"),
            np.load(FULL_RESIDUAL_PDP, mmap_mode="r"),
        )
    pas_output = np.lib.format.open_memmap(
        FULL_RESIDUAL_PAS,
        mode="w+",
        dtype=np.float16,
        shape=core_pas.shape,
    )
    pdp_output = np.lib.format.open_memmap(
        FULL_RESIDUAL_PDP,
        mode="w+",
        dtype=np.float16,
        shape=core_pdp.shape,
    )
    for start in range(0, len(core_pas), 32):
        end = min(start + 32, len(core_pas))
        pas_output[start:end] = (
            np.sqrt(np.maximum(np.asarray(truth_pas[start:end], dtype=np.float32), 0.0))
            - np.sqrt(np.maximum(np.asarray(core_pas[start:end], dtype=np.float32), 0.0))
        ).astype(np.float16)
        pdp_output[start:end] = (
            np.sqrt(np.maximum(np.asarray(truth_pdp[start:end], dtype=np.float32), 0.0))
            - np.sqrt(np.maximum(np.asarray(core_pdp[start:end], dtype=np.float32), 0.0))
        ).astype(np.float16)
    pas_output.flush()
    pdp_output.flush()
    return (
        np.load(FULL_RESIDUAL_PAS, mmap_mode="r"),
        np.load(FULL_RESIDUAL_PDP, mmap_mode="r"),
    )


def fit_basis(
    residual: np.ndarray,
    rows: np.ndarray,
    rank: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, float]:
    torch.manual_seed(seed)
    matrix = torch.as_tensor(
        np.asarray(residual[rows], dtype=np.float32).reshape(len(rows), -1),
        device=device,
    )
    keep = min(rank, len(rows) - 1, matrix.shape[1])
    _, singular, vectors = torch.pca_lowrank(
        matrix, q=keep, center=False, niter=4
    )
    basis = vectors.T.contiguous()
    projected_energy = singular.square().sum(dtype=torch.float64)
    total_energy = matrix.square().sum(dtype=torch.float64)
    explained = float((projected_energy / total_energy.clamp_min(TINY)).cpu())
    return basis.cpu().numpy().astype(np.float32), explained


def reconstruct(
    base: np.ndarray,
    truth: np.ndarray,
    basis: np.ndarray,
    probability_axes: tuple[int, ...],
) -> tuple[np.ndarray, float]:
    base_root = np.sqrt(np.maximum(np.asarray(base, dtype=np.float32), 0.0))
    truth_root = np.sqrt(np.maximum(np.asarray(truth, dtype=np.float32), 0.0))
    delta = (truth_root - base_root).reshape(len(base), -1)
    coefficient = delta @ basis.T
    reconstructed_delta = coefficient @ basis
    captured = float(
        np.sum(reconstructed_delta * reconstructed_delta, dtype=np.float64)
        / max(float(np.sum(delta * delta, dtype=np.float64)), TINY)
    )
    amplitude = np.maximum(
        base_root.reshape(len(base), -1) + reconstructed_delta, 0.0
    ).reshape(base.shape)
    power = amplitude * amplitude
    total = power.sum(axis=probability_axes, keepdims=True, dtype=np.float64)
    return (
        np.divide(
            power,
            np.maximum(total, TINY),
            out=np.zeros_like(power),
            where=total > 0.0,
        ),
        captured,
    )


def render_full_targets(
    pas: np.ndarray, pdp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    full_pas = np.repeat(pas[..., None], PAS_GROUP_SIZE, axis=-1)
    full_pas = full_pas.reshape(len(pas), 2, 16, 8, 4, 192)
    full_pdp = np.repeat(pdp, PDP_H_SIZE, axis=2)
    full_pdp = np.repeat(full_pdp, PDP_V_SIZE, axis=3)
    return (
        full_pas.reshape(len(pas), 256, 4, 192).astype(np.float32),
        full_pdp.reshape(len(pdp), 256, 4, 192).astype(np.float32),
    )


def score_channel(
    prediction: np.ndarray,
    truth: np.ndarray,
    query_ids: np.ndarray,
    batch_size: int,
) -> dict[str, float | int]:
    accumulator = OfficialMetricAccumulator()
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        accumulator.add(
            np.asarray(prediction[start:end], dtype=np.complex64),
            np.asarray(truth[query_ids[start:end]], dtype=np.complex64),
        )
    return accumulator.result()


def lifecycle_policy() -> tuple[str, dict[tuple[int, int, int], str]]:
    selection = json.loads(LIFECYCLE_SELECTION.read_text(encoding="utf-8"))
    winner = selection["winner"]
    stage = str(winner["stage"])
    policy = {
        (
            int(row["held_fold"]),
            int(row["site"]),
            REGIME_NAMES.index(str(row["regime"])),
        ): str(row["candidate"])
        for row in winner["nested_decisions"]
    }
    return stage, policy


@torch.inference_mode()
def current_ad(channel: torch.Tensor) -> torch.Tensor:
    shaped = channel.reshape(len(channel), 2, 16, 8, 4, 192)
    transformed = torch.fft.ifft(torch.fft.fft2(shaped, dim=(2, 3)), dim=-1)
    power = transformed.abs().square().sum(dim=(1, 4)).float()
    return normalize(power, (1, 2, 3))


def apply_lifecycle(
    source: np.ndarray,
    output_path,
    fold_id: np.ndarray,
    site: np.ndarray,
    regime: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    stage, policy = lifecycle_policy()
    target = np.load(TARGET_PATHS[stage][0], mmap_mode="r")
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.complex64, shape=source.shape
    )
    for start in range(0, len(source), batch_size):
        end = min(start + batch_size, len(source))
        block = np.array(source[start:end], dtype=np.complex64, copy=True)
        tensor = torch.as_tensor(block, device=device)
        current = current_ad(tensor)
        target_tensor = torch.as_tensor(
            np.asarray(target[start:end], dtype=np.float32), device=device
        )
        for fold_value in np.unique(fold_id[start:end]):
            for site_value in np.unique(site[start:end]):
                for regime_value in np.unique(regime[start:end]):
                    rows = np.flatnonzero(
                        (fold_id[start:end] == fold_value)
                        & (site[start:end] == site_value)
                        & (regime[start:end] == regime_value)
                    )
                    if len(rows) == 0:
                        continue
                    name = policy.get(
                        (int(fold_value), int(site_value), int(regime_value)),
                        "identity",
                    )
                    if name == "identity":
                        continue
                    prediction = apply_candidate(
                        tensor[rows],
                        current[rows],
                        target_tensor[rows],
                        regime[start:end][rows],
                        site[start:end][rows],
                        _candidate_from_name(name),
                    )
                    tensor[rows] = prediction
        output[start:end] = tensor.cpu().numpy().astype(np.complex64)
        output.flush()
    return np.load(output_path, mmap_mode="r")


def run_capacity(
    device: torch.device,
    batch_size: int,
    rank: int,
    cycles: int,
    force_endpoints: bool,
) -> dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    full_core = np.load(FULL_CORE, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    core_pas, core_pdp = build_endpoint_cache(
        full_core, "core", device, batch_size, force_endpoints
    )
    truth_pas, truth_pdp = build_endpoint_cache(
        truth_all, "truth", device, batch_size, force_endpoints
    )
    residual_pas, residual_pdp = build_residual_cache(
        core_pas, core_pdp, truth_pas, truth_pdp, force_endpoints
    )

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    canonical_fold = metadata["fold_id"].astype(np.int8)
    broad_fold_all = np.load(FOLD_IDS, allow_pickle=False).astype(np.int8)
    broad_fold = broad_fold_all[query_ids]
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    full_site = train_site.astype(np.int8)
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)
    canonical_core = np.load(OOF_SCALED, mmap_mode="r")

    canonical_core_pas, canonical_core_pdp = grouped_endpoints(
        torch.as_tensor(np.asarray(canonical_core), device=device)
    )
    canonical_truth = torch.as_tensor(
        np.asarray(truth_all[query_ids]), device=device
    )
    canonical_truth_pas, canonical_truth_pdp = grouped_endpoints(canonical_truth)
    canonical_core_pas_np = canonical_core_pas.cpu().numpy()
    canonical_core_pdp_np = canonical_core_pdp.cpu().numpy()
    canonical_truth_pas_np = canonical_truth_pas.cpu().numpy()
    canonical_truth_pdp_np = canonical_truth_pdp.cpu().numpy()
    target_pas = np.empty_like(canonical_core_pas_np)
    target_pdp = np.empty_like(canonical_core_pdp_np)
    basis_records: list[dict[str, object]] = []

    for outer in range(5):
        for site_value in (0, 1):
            fit_rows = np.flatnonzero(
                (broad_fold_all != outer) & (full_site == site_value)
            )
            held_rows = np.flatnonzero(
                (broad_fold == outer) & (site == site_value)
            )
            if len(held_rows) == 0:
                continue
            pas_basis, pas_explained = fit_basis(
                residual_pas,
                fit_rows,
                rank,
                device,
                20_260_810 + outer * 101 + site_value,
            )
            pdp_basis, pdp_explained = fit_basis(
                residual_pdp,
                fit_rows,
                rank,
                device,
                20_270_810 + outer * 101 + site_value,
            )
            target_pas[held_rows], pas_captured = reconstruct(
                canonical_core_pas_np[held_rows],
                canonical_truth_pas_np[held_rows],
                pas_basis,
                (1, 2, 3),
            )
            target_pdp[held_rows], pdp_captured = reconstruct(
                canonical_core_pdp_np[held_rows],
                canonical_truth_pdp_np[held_rows],
                pdp_basis,
                (5,),
            )
            basis_records.append(
                {
                    "broad_fold": outer,
                    "site": site_value,
                    "fit_count": int(len(fit_rows)),
                    "held_count": int(len(held_rows)),
                    "pas_explained": pas_explained,
                    "pdp_explained": pdp_explained,
                    "pas_held_captured": pas_captured,
                    "pdp_held_captured": pdp_captured,
                }
            )
            print(
                f"E160 basis outer={outer} site={site_value} "
                f"explained={pas_explained:.4f}/{pdp_explained:.4f} "
                f"held={pas_captured:.4f}/{pdp_captured:.4f}",
                flush=True,
            )

    full_target_pas, full_target_pdp = render_full_targets(target_pas, target_pdp)
    oracle_path = ARTIFACT_DIR / f"canonical_oracle_rank{rank}_pre_lifecycle.npy"
    complete_path = ARTIFACT_DIR / f"canonical_oracle_rank{rank}_complete.npy"
    oracle = np.lib.format.open_memmap(
        oracle_path,
        mode="w+",
        dtype=np.complex64,
        shape=canonical_core.shape,
    )
    for start in range(0, len(canonical_core), batch_size):
        end = min(start + batch_size, len(canonical_core))
        base = torch.as_tensor(
            np.array(canonical_core[start:end], dtype=np.complex64, copy=True),
            device=device,
        )
        pas = torch.as_tensor(full_target_pas[start:end], device=device).reshape(
            end - start, 2, 16, 8, 4, 192
        )
        pdp = torch.as_tensor(full_target_pdp[start:end], device=device)
        reference_power = base.abs().square().sum(dim=(1, 2, 3))
        block = base
        for _ in range(cycles):
            for site_value in np.unique(site[start:end]):
                rows = np.flatnonzero(site[start:end] == site_value)
                local = block[rows]
                if int(site_value) == 0:
                    local = exact_pdp_projection(local, pdp[rows])
                    local = exact_pas_projection(local, pas[rows])
                else:
                    local = exact_pas_projection(local, pas[rows])
                    local = exact_pdp_projection(local, pdp[rows])
                block[rows] = preserve_point_power(local, reference_power[rows])
        oracle[start:end] = block.cpu().numpy().astype(np.complex64)
        oracle.flush()
    pre_score = score_channel(oracle, truth_all, query_ids, batch_size)
    complete = apply_lifecycle(
        oracle,
        complete_path,
        canonical_fold,
        site,
        regime,
        device,
        batch_size,
    )
    complete_score = score_channel(complete, truth_all, query_ids, batch_size)
    return {
        "rank": rank,
        "cycles": cycles,
        "required_score": REQUIRED_SCORE,
        "passed": float(complete_score["score"]) > REQUIRED_SCORE,
        "pre_lifecycle": {
            "path": PATHS.relative(oracle_path),
            "official": pre_score,
        },
        "complete": {
            "path": PATHS.relative(complete_path),
            "official": complete_score,
        },
        "basis": basis_records,
        "residuals": {
            "pas": PATHS.relative(FULL_RESIDUAL_PAS),
            "pdp": PATHS.relative(FULL_RESIDUAL_PDP),
        },
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    result = run_capacity(
        torch.device(args.device),
        args.batch_size,
        args.rank,
        args.cycles,
        args.force_endpoints,
    )
    payload = {
        "experiment": VERSION,
        "status": "capacity_passed" if result["passed"] else "capacity_failed",
        "official_inputs_only": True,
        "sionna_used": False,
        "oracle_not_deployable": True,
        "result": result,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
