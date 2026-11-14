#!/usr/bin/env python3
"""Deploy the five strict E190 path calibrators to the official Test points."""

from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np
import torch

import analyze_e86_surface_endpoint_v11_blend as e86
import generate_e86_surface_endpoint_test_channel as e86_test
import optimize_e98_global_array_angle_mapping as e98
import train_e190_official_sionna_path_calibrator as e190
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E190-official-sionna-path-calibrator-test"
ARTIFACT_DIR = e190.ARTIFACT_DIR
ANALYSIS_DIR = e190.ANALYSIS_DIR
OUTPUT = ARTIFACT_DIR / "Round2_E190_Official_Sionna_Path_Calibrator_Test_Channel.npy"
ENDPOINT = ARTIFACT_DIR / "test_fivefold_official_path_calibrator_endpoint.npz"
REPORT = ANALYSIS_DIR / "test_generate.json"
PERIODS = np.asarray((e85.MH, e85.MV, e85.S_SC), dtype=np.float32)
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def required_inputs() -> tuple:
    checkpoints = tuple(e190.refit_checkpoint_path(outer) for outer in range(5))
    return (
        e86_test.TEST_PATH_CACHE,
        e86_test.TEST_FEATURES,
        e86_test.SIONNA_STATE,
        e86_test.V11_TEST,
        e86_test.LIGHT_ENDPOINT_OUTPUT,
        e86_test.OUTPUT,
        e190.FULL_REPORT,
        *checkpoints,
    )


def make_model(
    checkpoint: dict[str, object],
    feature_dim: int,
    sequence_dim: int,
    device: torch.device,
) -> e85.SurfaceInteractionModel:
    config = e190.Config(**checkpoint["config"])
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
    model = e85.SurfaceInteractionModel(
        feature_dim, sequence_dim, model_config, "geometry"
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def mapped_test_coordinates(
    auxiliary: dict[str, np.ndarray],
    latent: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    _, _, _, _, test_site = load_positions()
    angles = np.array(
        np.load(e86_test.TEST_PATH_CACHE / "angles.npy", mmap_mode="r"),
        dtype=np.float32,
        copy=True,
    )
    parameters = e98.decode_parameters(
        torch.as_tensor(latent, dtype=torch.float32, device=device)
    )
    site = torch.as_tensor(test_site.astype(np.int64), device=device)
    theta = torch.as_tensor(angles[..., 0], device=device)
    phi = torch.as_tensor(angles[..., 1], device=device)
    yaw = parameters["yaw"][site][:, None]
    tilt = parameters["tilt"][site][:, None]
    horizontal = (
        e85.MH
        * parameters["horizontal_spacing"]
        * torch.sin(theta)
        * torch.sin(phi - yaw)
        + parameters["horizontal_offset"][site][:, None]
    )
    vertical = (
        e85.MV
        * parameters["vertical_spacing"]
        * (
            torch.sin(tilt) * torch.sin(theta) * torch.cos(phi - yaw)
            - torch.cos(tilt) * torch.cos(theta)
        )
        + parameters["vertical_offset"][site][:, None]
    )
    coordinate = np.array(auxiliary["coordinate"], dtype=np.float32, copy=True)
    coordinate[..., 0] = torch.remainder(horizontal, float(e85.MH)).cpu().numpy()
    coordinate[..., 1] = torch.remainder(vertical, float(e85.MV)).cpu().numpy()
    return coordinate


def circular_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    phase = values * (2.0 * np.pi / PERIODS)[None, None, None, :]
    expanded = weights[..., None]
    sine = np.sum(expanded * np.sin(phase), axis=0, dtype=np.float64)
    cosine = np.sum(expanded * np.cos(phase), axis=0, dtype=np.float64)
    angle = np.mod(np.arctan2(sine, cosine), 2.0 * np.pi)
    return (angle * PERIODS / (2.0 * np.pi)).astype(np.float32)


@torch.inference_mode()
def predict_ensemble(
    features: np.ndarray,
    auxiliary: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    rows = np.arange(len(features), dtype=np.int64)
    fold_outputs: list[dict[str, np.ndarray]] = []
    fold_reports: list[dict[str, object]] = []
    for outer in range(5):
        checkpoint_path = e190.refit_checkpoint_path(outer)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if int(checkpoint["outer"]) != outer:
            raise RuntimeError(f"E190 outer identity mismatch in {checkpoint_path}")
        model = make_model(
            checkpoint, features.shape[-1], auxiliary["geometry"].shape[-1], device
        )
        fold_auxiliary = dict(auxiliary)
        fold_auxiliary["coordinate"] = mapped_test_coordinates(
            auxiliary, np.asarray(checkpoint["mapping_latent"], dtype=np.float32), device
        )
        mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        local_outputs = []
        for start in range(0, len(rows), batch_size):
            local = rows[start : start + batch_size]
            local_outputs.append(
                model(e86_test.test_batch(local, features, fold_auxiliary, mean, std, device))
            )
        prediction = {
            key: np.concatenate(
                [value[key].detach().cpu().numpy() for value in local_outputs], axis=0
            )
            for key in ("mass", "port", "coordinate")
        }
        prediction["coordinate"] = np.mod(prediction["coordinate"], PERIODS)
        fold_outputs.append(prediction)
        fold_reports.append(
            {
                "outer": outer,
                "best_epoch": int(checkpoint["best_epoch"]),
                "checkpoint": PATHS.relative(checkpoint_path),
            }
        )
        del model, local_outputs, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"E190 Test inference fold {outer + 1}/5", flush=True)

    mass_stack = np.stack([value["mass"] for value in fold_outputs], axis=0)
    port_stack = np.stack([value["port"] for value in fold_outputs], axis=0)
    coordinate_stack = np.stack(
        [value["coordinate"] for value in fold_outputs], axis=0
    )
    mass = mass_stack.mean(axis=0, dtype=np.float64)
    mass /= np.maximum(mass.sum(axis=1, keepdims=True), TINY)
    port = port_stack.mean(axis=0, dtype=np.float64)
    port /= np.maximum(port.sum(axis=(2, 3), keepdims=True), TINY)
    coordinate_weight = mass_stack / np.maximum(
        mass_stack.sum(axis=0, keepdims=True), TINY
    )
    coordinate = circular_mean(coordinate_stack, coordinate_weight)
    agreement = {
        "mass_fold_std_mean": float(np.mean(np.std(mass_stack, axis=0))),
        "port_fold_std_mean": float(np.mean(np.std(port_stack, axis=0))),
        "coordinate_circular_resultant_mean": float(
            np.mean(
                np.abs(
                    np.sum(
                        coordinate_weight[..., None]
                        * np.exp(
                            2j
                            * np.pi
                            * coordinate_stack
                            / PERIODS[None, None, None, :]
                        ),
                        axis=0,
                    )
                )
            )
        ),
    }
    fold_reports.append({"ensemble_agreement": agreement})
    return {
        "mass": mass.astype(np.float32),
        "coordinate": coordinate.astype(np.float32),
        "port": port.astype(np.float32),
    }, fold_reports


@torch.inference_mode()
def project_endpoint(
    base: torch.Tensor,
    endpoint_pas: np.ndarray,
    endpoint_pdp: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    spec = e86.Candidate(
        endpoint="official_path_calibrator",
        mode="sqrt",
        pas_beta=0.20,
        pdp_beta=0.20,
        order="pas_pdp",
        cycles=4,
    )
    return e86.make_candidate(
        base,
        pas_distribution(base),
        pdp_distribution(base),
        torch.as_tensor(endpoint_pas, device=device),
        torch.as_tensor(endpoint_pdp, device=device),
        spec,
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    phase_time: dict[str, float] = {}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    missing = [PATHS.relative(path) for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing E190 Test inputs: {missing}")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    load_start = time.time()
    features = np.load(e86_test.TEST_FEATURES, mmap_mode="r")
    auxiliary = e86_test.build_test_auxiliary()
    base_np = np.load(e86_test.V11_TEST, mmap_mode="r")
    base = torch.as_tensor(np.array(base_np, dtype=np.complex64, copy=True), device=device)
    phase_time["load_test_inputs"] = float(time.time() - load_start)

    identity_start = time.time()
    old_endpoint = np.load(e86_test.LIGHT_ENDPOINT_OUTPUT, allow_pickle=False)
    old_pas, old_pdp = e190.render_endpoint(
        old_endpoint["mass"].astype(np.float32),
        old_endpoint["coordinate"].astype(np.float32),
        old_endpoint["port"].astype(np.float32),
        device,
    )
    identity = project_endpoint(base, old_pas, old_pdp, device)
    old_output = torch.as_tensor(
        np.array(np.load(e86_test.OUTPUT, mmap_mode="r"), copy=True), device=device
    )
    identity_error = float(torch.max(torch.abs(identity - old_output)).item())
    if identity_error > 1e-6:
        raise RuntimeError(f"V12 Test projection identity failed: {identity_error}")
    del identity, old_output, old_pas, old_pdp, old_endpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    phase_time["v12_test_identity"] = float(time.time() - identity_start)

    inference_start = time.time()
    prediction, fold_reports = predict_ensemble(
        features, auxiliary, device, args.batch_size
    )
    phase_time["fivefold_inference"] = float(time.time() - inference_start)

    render_start = time.time()
    endpoint_pas, endpoint_pdp = e190.render_endpoint(
        prediction["mass"], prediction["coordinate"], prediction["port"], device
    )
    phase_time["endpoint_render"] = float(time.time() - render_start)

    projection_start = time.time()
    output = project_endpoint(base, endpoint_pas, endpoint_pdp, device)
    output_np = output.cpu().numpy().astype(np.complex64)
    np.save(OUTPUT, output_np)
    np.savez_compressed(
        ENDPOINT,
        mass=prediction["mass"],
        coordinate=prediction["coordinate"],
        port=prediction["port"],
    )
    power_ratio = (
        output.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
        / base.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64).clamp_min(TINY)
    ).cpu().numpy()
    phase_time["projection_and_save"] = float(time.time() - projection_start)

    fixed = json.loads(e190.FULL_REPORT.read_text(encoding="utf-8"))
    peak_memory = (
        float(torch.cuda.max_memory_allocated() / 2**30)
        if device.type == "cuda"
        else 0.0
    )
    report = {
        "experiment": VERSION,
        "status": "test_candidate_generated",
        "data_lineage": {
            "official_train_h_used_to_train_checkpoints": True,
            "official_test_positions_used": True,
            "same_position_test_sionna_paths_used": True,
            "hidden_official_test_h_used": False,
            "absolute_sionna_phase_used": False,
            "v11_complex_carrier_preserved": True,
        },
        "local_validation": {
            "complete_fixed500_score": fixed["selected"]["score"],
            "v12_complete_fixed500_score": fixed["baseline"]["score"],
            "delta": fixed["delta"]["score"],
            "original_deployment_gate_passed": fixed["gate"]["passed"],
            "user_authorized_failed_gate_deployment": True,
        },
        "deployment_policy": {
            "models": "mean of five strict E190 refit checkpoints",
            "mass": "arithmetic mean then normalize",
            "port": "arithmetic mean then normalize",
            "coordinate": "per-candidate mass-weighted circular mean",
            "pas_beta": 0.20,
            "pdp_beta": 0.20,
            "mode": "sqrt",
            "order": "pas_pdp",
            "cycles": 4,
        },
        "folds": fold_reports,
        "identity": {"v12_test_projection_max_abs_error": identity_error},
        "output": {
            "path": PATHS.relative(OUTPUT),
            "shape": list(output_np.shape),
            "dtype": str(output_np.dtype),
            "finite": bool(np.isfinite(output_np).all()),
            "bytes": int(OUTPUT.stat().st_size),
            "power_ratio_mean": float(np.mean(power_ratio)),
            "power_ratio_max_abs_error": float(np.max(np.abs(power_ratio - 1.0))),
        },
        "endpoint": {
            "path": PATHS.relative(ENDPOINT),
            "mass_shape": list(prediction["mass"].shape),
            "coordinate_shape": list(prediction["coordinate"].shape),
            "port_shape": list(prediction["port"].shape),
            "finite": bool(
                np.isfinite(prediction["mass"]).all()
                and np.isfinite(prediction["coordinate"]).all()
                and np.isfinite(prediction["port"]).all()
            ),
        },
        "performance": {
            "device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
            "peak_torch_memory_gib": peak_memory,
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "timing_seconds": phase_time,
            "measured_bottleneck": max(phase_time, key=phase_time.get),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
