#!/usr/bin/env python3
"""Evaluate one frozen full-path joint-domain generator on canonical Round 2 holes."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from analyze_e86_surface_endpoint_v11_blend import endpoint_target
from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
    truth_views,
)
from analyze_v4_path_track_spectral_targets import spectral_kernels
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics
from train_full_path_pas_slice_gate import project_joint


VERSION = "Round2-E189-end-to-end-complex-path-fusion"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e189_end_to_end_complex_path_fusion"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_e189_end_to_end_complex_path_fusion"
)
REPORT = ANALYSIS_DIR / "fixed500_report.json"
OUTPUT_CHANNEL = ARTIFACT_DIR / "e189_fixed500_channel.npy"

OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
V11_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "canonical_500_version_replay"
    / "V11_Canonical_500_OOF_Channel.npy"
)
V12_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_frozen_geometry_sequence_oof_channel.npy"
)
ENDPOINT = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "strict_oof_geometry_sequence_path_endpoint.npz"
)
ENDPOINT_PAS = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_geometry_sequence_rendered_pas_f16.npy"
)
ENDPOINT_PDP = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_geometry_sequence_rendered_pdp_f16.npy"
)
PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n4000_s24_k128_b096_d064_scatter1_sel106"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
TINY = 1e-30

# Frozen from the E86 policy and the earlier joint-domain Sionna screen.
MARGINAL_BETA = 0.20
JOINT_BETA_MIN = 0.05
JOINT_BETA_MAX = 0.10
COHERENCE_MASS_FLOOR = 0.75
CYCLES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def normalized(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), 0.0)


def joint_distribution(channel: torch.Tensor) -> torch.Tensor:
    shaped = channel.reshape(len(channel), MP, MH, MV, N_UE, S_SC)
    transformed = torch.fft.ifft(
        torch.fft.fft2(shaped, dim=(2, 3)), dim=5
    )
    return normalized(transformed.abs().square(), (1, 2, 3, 4, 5))


def exact_joint_projection(
    channel: torch.Tensor, desired: torch.Tensor
) -> torch.Tensor:
    shaped = channel.reshape(len(channel), MP, MH, MV, N_UE, S_SC)
    transformed = torch.fft.ifft(
        torch.fft.fft2(shaped, dim=(2, 3)), dim=5
    )
    magnitude = transformed.abs()
    total = magnitude.square().sum(dim=(1, 2, 3, 4, 5), keepdim=True)
    desired_magnitude = torch.sqrt((desired * total).clamp_min(0.0))
    unit = transformed / magnitude.clamp_min(TINY)
    unit = torch.where(
        magnitude > TINY,
        unit,
        torch.ones((), dtype=transformed.dtype, device=transformed.device),
    )
    restored = torch.fft.ifft2(
        torch.fft.fft(desired_magnitude * unit, dim=5), dim=(2, 3)
    )
    return restored.reshape_as(channel)


def path_array_coherence(
    coefficient: torch.Tensor,
    coordinate: torch.Tensor,
    port: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Measure whether each full complex path supports its calibrated H/V coordinate."""

    shaped = coefficient.reshape(len(coefficient), -1, MP, MH, MV, N_UE)
    horizontal = torch.arange(MH, dtype=torch.float32, device=coefficient.device)
    vertical = torch.arange(MV, dtype=torch.float32, device=coefficient.device)
    phase = 2.0 * torch.pi * (
        coordinate[..., 0, None, None] * horizontal[None, None, :, None] / MH
        + coordinate[..., 1, None, None] * vertical[None, None, None, :] / MV
    )
    steering = torch.exp(1j * phase)
    inner = torch.einsum(
        "bkphvu,bkhv->bkpu", shaped, steering.conj()
    )
    energy = shaped.abs().square().sum(dim=(3, 4))
    denominator = torch.sqrt((energy * float(MH * MV)).clamp_min(TINY))
    per_port = (inner.abs() / denominator).clamp(0.0, 1.0)
    coherence = (per_port * port).sum(dim=(2, 3))
    return coherence * mask.to(coherence.dtype)


def render_joint_target(
    mass: torch.Tensor,
    coordinate: torch.Tensor,
    port: torch.Tensor,
) -> torch.Tensor:
    horizontal, vertical, delay = spectral_kernels(coordinate)
    energy = mass[..., None, None] * port
    target = torch.einsum(
        "bkpu,bkh,bkv,bkd->bphvud",
        energy,
        horizontal,
        vertical,
        delay,
    )
    return normalized(target, (1, 2, 3, 4, 5))


def sqrt_blend_joint(
    current: torch.Tensor,
    target: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    mixed = (
        (1.0 - beta) * torch.sqrt(current.clamp_min(0.0))
        + beta * torch.sqrt(target.clamp_min(0.0))
    ).square()
    return normalized(mixed, (1, 2, 3, 4, 5))


def joint_marginals(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pas = target.sum(dim=5)
    pas = normalized(pas, (1, 2, 3))
    pdp = target.sum(dim=(2, 3))
    pdp = normalized(pdp, (3,))
    return pas, pdp


def make_joint_candidate(
    base: torch.Tensor,
    desired_joint: torch.Tensor,
    desired_pas: torch.Tensor,
    desired_pdp: torch.Tensor,
) -> torch.Tensor:
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    output = base
    for _ in range(CYCLES):
        output = exact_joint_projection(output, desired_joint)
        output = exact_pas_projection(output, desired_pas)
        output = exact_pdp_projection(output, desired_pdp)
        output = preserve_point_power(output, reference_power)
    return output


def allocate_metrics(count: int) -> tuple[np.ndarray, ...]:
    return tuple(np.empty(count, dtype=np.float64) for _ in range(4))


def assign_metrics(
    destination: tuple[np.ndarray, ...],
    rows: slice,
    source: tuple[np.ndarray, ...],
) -> None:
    for output, value in zip(destination, source):
        output[rows] = value


def summarize(
    values: tuple[np.ndarray, ...], fold_id: np.ndarray
) -> dict[str, object]:
    overall = statistics(*values)
    return {
        "overall": overall,
        "folds": [
            {"fold": fold, **statistics(*values, fold_id == fold)}
            for fold in range(5)
        ],
    }


def component_deltas(
    selected: dict[str, float | int], baseline: dict[str, float | int]
) -> dict[str, float]:
    return {
        "pas": float(selected["pas"] - baseline["pas"]),
        "pdp": float(selected["pdp"] - baseline["pdp"]),
        "nmse_score": float(selected["nmse_score"] - baseline["nmse_score"]),
        "score": float(selected["score"] - baseline["score"]),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("E189 requires CUDA for the fixed500 complete-H screen")
    device_index = 0 if device.index is None else device.index
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    load_started = time.time()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    endpoint = np.load(ENDPOINT, allow_pickle=False)
    if not np.array_equal(endpoint["query_ids"].astype(np.int64), query_ids):
        raise ValueError("E85 endpoint/query identity mismatch")
    path_rows = np.load(PATH_CACHE / "rows.npy", allow_pickle=False)
    if not np.array_equal(path_rows, np.arange(4000, dtype=np.int64)):
        raise ValueError("full path cache rows are not the official 0..3999 order")

    v11 = np.load(V11_CHANNEL, mmap_mode="r")
    v12 = np.load(V12_CHANNEL, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    endpoint_pas = np.load(ENDPOINT_PAS, mmap_mode="r")
    endpoint_pdp = np.load(ENDPOINT_PDP, mmap_mode="r")
    coefficient_all = np.load(PATH_CACHE / "coefficient.npy", mmap_mode="r")
    mask_all = np.load(PATH_CACHE / "mask.npy", mmap_mode="r")
    coefficient = np.asarray(coefficient_all[query_ids], dtype=np.complex64)
    path_mask = np.asarray(mask_all[query_ids], dtype=bool)
    load_seconds = time.time() - load_started

    expected_shape = (len(query_ids), 256, 4, 192)
    for name, value in (("V11", v11), ("V12", v12)):
        if value.shape != expected_shape or value.dtype != np.complex64:
            raise ValueError(f"{name} shape/dtype mismatch: {value.shape}/{value.dtype}")

    output = np.lib.format.open_memmap(
        OUTPUT_CHANNEL,
        mode="w+",
        dtype=np.complex64,
        shape=expected_shape,
    )
    metric_names = (
        "v11",
        "v12",
        "marginal_identity_replay",
        "joint_without_complex_coherence",
        "e189_complex_coherence",
    )
    metrics = {name: allocate_metrics(len(query_ids)) for name in metric_names}

    identity_max_abs = 0.0
    identity_power_max_rel = 0.0
    endpoint_pas_max_abs = 0.0
    endpoint_pdp_max_abs = 0.0
    output_power_max_rel = 0.0
    point_coherence = np.empty(len(query_ids), dtype=np.float32)
    joint_beta = np.empty(len(query_ids), dtype=np.float32)
    stage_seconds = {"target": 0.0, "projection": 0.0, "scoring": 0.0}

    for start in range(0, len(query_ids), args.batch_size):
        end = min(start + args.batch_size, len(query_ids))
        rows = slice(start, end)
        base = torch.as_tensor(
            np.array(v11[rows], dtype=np.complex64, copy=True), device=device
        )
        baseline_v12 = torch.as_tensor(
            np.array(v12[rows], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(
                truth_all[query_ids[rows]], dtype=np.complex64, copy=True
            ),
            device=device,
        )
        mass = torch.as_tensor(endpoint["mass"][rows], device=device)
        coordinate = torch.as_tensor(endpoint["coordinate"][rows], device=device)
        port = torch.as_tensor(endpoint["port"][rows], device=device)
        local_mask = torch.as_tensor(path_mask[rows], device=device)
        local_coefficient = torch.as_tensor(coefficient[rows], device=device)
        rendered_pas = torch.as_tensor(
            np.asarray(endpoint_pas[rows], dtype=np.float32), device=device
        )
        rendered_pdp = torch.as_tensor(
            np.asarray(endpoint_pdp[rows], dtype=np.float32), device=device
        )

        target_started = time.time()
        with torch.inference_mode():
            base_pas = pas_distribution(base)
            base_pdp = pdp_distribution(base)
            desired_pas = endpoint_target(
                base_pas,
                rendered_pas,
                MARGINAL_BETA,
                "sqrt",
                (1, 2, 3),
            )
            desired_pdp = endpoint_target(
                base_pdp,
                rendered_pdp,
                MARGINAL_BETA,
                "sqrt",
                (3,),
            )

            raw_joint = render_joint_target(mass, coordinate, port)
            raw_pas, raw_pdp = joint_marginals(raw_joint)
            endpoint_pas_max_abs = max(
                endpoint_pas_max_abs,
                float((raw_pas - rendered_pas[..., 0]).abs().max().item()),
            )
            rendered_pdp_small = rendered_pdp.reshape(
                end - start, MP, MH, MV, N_UE, S_SC
            )[:, :, 0, 0]
            endpoint_pdp_max_abs = max(
                endpoint_pdp_max_abs,
                float((raw_pdp - rendered_pdp_small).abs().max().item()),
            )

            coherence = path_array_coherence(
                local_coefficient, coordinate, port, local_mask
            )
            weighted_coherence = (
                (mass * coherence).sum(dim=1)
                / mass.sum(dim=1).clamp_min(TINY)
            ).clamp(0.0, 1.0)
            calibrated_mass = mass * (
                COHERENCE_MASS_FLOOR
                + (1.0 - COHERENCE_MASS_FLOOR) * coherence
            )
            calibrated_mass = calibrated_mass * local_mask.to(calibrated_mass.dtype)
            calibrated_mass = calibrated_mass / calibrated_mass.sum(
                dim=1, keepdim=True
            ).clamp_min(TINY)
            calibrated_joint = render_joint_target(
                calibrated_mass, coordinate, port
            )
            current_joint = joint_distribution(base)
            local_beta = (
                JOINT_BETA_MIN
                + (JOINT_BETA_MAX - JOINT_BETA_MIN) * weighted_coherence
            ).reshape(-1, 1, 1, 1, 1, 1)
            desired_joint = sqrt_blend_joint(
                current_joint, calibrated_joint, local_beta
            )
            control_beta = torch.full_like(local_beta, JOINT_BETA_MAX)
            control_joint = sqrt_blend_joint(
                current_joint, raw_joint, control_beta
            )
        stage_seconds["target"] += time.time() - target_started

        projection_started = time.time()
        with torch.inference_mode():
            identity = project_joint(
                base, desired_pas, desired_pdp, CYCLES, "pas_pdp"
            )
            control = make_joint_candidate(
                base, control_joint, desired_pas, desired_pdp
            )
            candidate = make_joint_candidate(
                base, desired_joint, desired_pas, desired_pdp
            )
        stage_seconds["projection"] += time.time() - projection_started

        identity_max_abs = max(
            identity_max_abs,
            float((identity - baseline_v12).abs().max().item()),
        )
        identity_power = identity.abs().square().sum(dim=(1, 2, 3))
        v12_power = baseline_v12.abs().square().sum(dim=(1, 2, 3))
        identity_power_max_rel = max(
            identity_power_max_rel,
            float(
                (
                    (identity_power - v12_power).abs()
                    / v12_power.clamp_min(TINY)
                )
                .max()
                .item()
            ),
        )
        base_power = base.abs().square().sum(dim=(1, 2, 3))
        candidate_power = candidate.abs().square().sum(dim=(1, 2, 3))
        output_power_max_rel = max(
            output_power_max_rel,
            float(
                (
                    (candidate_power - base_power).abs()
                    / base_power.clamp_min(TINY)
                )
                .max()
                .item()
            ),
        )
        point_coherence[rows] = weighted_coherence.cpu().numpy().astype(np.float32)
        joint_beta[rows] = local_beta[:, 0, 0, 0, 0, 0].cpu().numpy().astype(np.float32)
        output[rows] = candidate.cpu().numpy().astype(np.complex64)

        scoring_started = time.time()
        with torch.inference_mode():
            truth_pas, truth_pdp = truth_views(truth)
            for name, prediction in (
                ("v11", base),
                ("v12", baseline_v12),
                ("marginal_identity_replay", identity),
                ("joint_without_complex_coherence", control),
                ("e189_complex_coherence", candidate),
            ):
                assign_metrics(
                    metrics[name],
                    rows,
                    point_metrics(prediction, truth, truth_pas, truth_pdp),
                )
        stage_seconds["scoring"] += time.time() - scoring_started
        print(f"E189 fixed500 {end}/{len(query_ids)}", flush=True)

    output.flush()
    summaries = {name: summarize(values, fold_id) for name, values in metrics.items()}
    selected = summaries["e189_complex_coherence"]["overall"]
    baseline = summaries["v12"]["overall"]
    fold_deltas = [
        float(
            selected_fold["score"] - baseline_fold["score"]
        )
        for selected_fold, baseline_fold in zip(
            summaries["e189_complex_coherence"]["folds"],
            summaries["v12"]["folds"],
        )
    ]
    identity_score_error = abs(
        float(
            summaries["marginal_identity_replay"]["overall"]["score"]
            - baseline["score"]
        )
    )
    identity_passed = bool(
        identity_max_abs <= 5e-5 and identity_score_error <= 5e-6
    )
    selected_deltas = component_deltas(selected, baseline)
    gate = {
        "minimum_score": 0.685,
        "minimum_positive_folds": 4,
        "maximum_component_drop": 0.001,
        "identity_passed": identity_passed,
        "score_passed": bool(selected["score"] >= 0.685),
        "positive_folds": int(sum(delta > 0.0 for delta in fold_deltas)),
        "fold_passed": bool(sum(delta > 0.0 for delta in fold_deltas) >= 4),
        "component_passed": bool(
            selected_deltas["pas"] >= -0.001
            and selected_deltas["pdp"] >= -0.001
            and selected_deltas["nmse_score"] >= -0.001
        ),
    }
    gate["passed"] = bool(
        gate["identity_passed"]
        and gate["score_passed"]
        and gate["fold_passed"]
        and gate["component_passed"]
    )

    elapsed = time.time() - started
    output_array = np.load(OUTPUT_CHANNEL, mmap_mode="r")
    payload = {
        "experiment": VERSION,
        "status": "fixed500_passed" if gate["passed"] else "fixed500_failed",
        "contract": {
            "type": "score_solution",
            "online_goal": 0.70,
            "fixed500_score_gate": 0.685,
            "no_parameter_scan": True,
        },
        "method": {
            "base": str(PATHS.relative(V11_CHANNEL)),
            "endpoint": str(PATHS.relative(ENDPOINT)),
            "external_path_cache": str(PATHS.relative(PATH_CACHE)),
            "marginal_beta": MARGINAL_BETA,
            "joint_beta_range": [JOINT_BETA_MIN, JOINT_BETA_MAX],
            "coherence_mass_floor": COHERENCE_MASS_FLOOR,
            "cycles": CYCLES,
            "order": "joint_pas_pdp",
            "absolute_sionna_phase_copied": False,
            "full_complex_coefficient_use": (
                "array-relative phase coherence against each calibrated path H/V "
                "coordinate; it calibrates path mass and joint-domain strength"
            ),
            "point_power_preserved": True,
        },
        "identity": {
            "query_count": int(len(query_ids)),
            "query_ids_sha256": hashlib.sha256(query_ids.tobytes()).hexdigest(),
            "endpoint_ids_equal": True,
            "path_rows_official_order": True,
            "marginal_replay_max_abs_h": identity_max_abs,
            "marginal_replay_score_error": identity_score_error,
            "marginal_replay_power_max_relative_error": identity_power_max_rel,
            "endpoint_joint_pas_max_abs": endpoint_pas_max_abs,
            "endpoint_joint_pdp_max_abs": endpoint_pdp_max_abs,
            "output_power_max_relative_error": output_power_max_rel,
            "passed": identity_passed,
        },
        "summaries": summaries,
        "selected_delta_vs_v12": selected_deltas,
        "fold_deltas_vs_v12": fold_deltas,
        "coherence": {
            "mean": float(np.mean(point_coherence)),
            "median": float(np.median(point_coherence)),
            "p10": float(np.quantile(point_coherence, 0.10)),
            "p90": float(np.quantile(point_coherence, 0.90)),
            "joint_beta_mean": float(np.mean(joint_beta)),
        },
        "gate": gate,
        "output": {
            "path": str(PATHS.relative(OUTPUT_CHANNEL)),
            "shape": list(output_array.shape),
            "dtype": str(output_array.dtype),
            "finite": bool(np.isfinite(output_array).all()),
        },
        "performance": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device_index),
            "peak_gpu_memory_gib": float(
                torch.cuda.max_memory_allocated(device_index) / (1024**3)
            ),
            "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2),
            "load_seconds": float(load_seconds),
            "target_seconds": float(stage_seconds["target"]),
            "projection_seconds": float(stage_seconds["projection"]),
            "scoring_seconds": float(stage_seconds["scoring"]),
            "wall_seconds": float(elapsed),
        },
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "v11": summaries["v11"]["overall"],
                "v12": baseline,
                "joint_control": summaries["joint_without_complex_coherence"]["overall"],
                "e189": selected,
                "delta_vs_v12": selected_deltas,
                "fold_deltas_vs_v12": fold_deltas,
                "gate": gate,
                "performance": payload["performance"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
