#!/usr/bin/env python3
"""Generate the E86 frozen-policy test channel from strict E85 models."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

import analyze_path_candidate_information_audit as e84
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from analyze_v4_path_track_spectral_targets import render_noncoherent
from data import load_positions
from project_paths import PATHS
from train_surface_interaction_energy_model import (
    ARTIFACT_DIR as MODEL_DIR,
    Config,
    SurfaceInteractionModel,
    sequence_geometry_block,
)
from train_full_path_pas_slice_gate import project_joint


VERSION = "Round2-E86-surface-endpoint-v11-blend"
TEST_PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "test_n500_s24_k128_b096_d064_scatter1_sel106"
)
TEST_FEATURES = PATHS.artifacts / "round2" / "explicit_path_pdp_predictor" / "path_features_test_n500.npy"
SIONNA_STATE = (
    PATHS.artifacts
    / "round2"
    / "sionna_multiseed"
    / "primary_d16_base_v4lambda_1m"
    / "full"
    / "state.json"
)
V11_TEST = (
    PATHS.artifacts
    / "round2"
    / "reliable_sqrt_pas_gate"
    / "Round2_V11_Reliable_Sqrt_PAS_Consensus_Test_Channel.npy"
)
OUTPUT_DIR = PATHS.artifacts / "round2" / "surface_interaction_energy_model"
ENDPOINT_OUTPUT = OUTPUT_DIR / "e86_test_geometry_sequence_endpoint.npz"
LIGHT_ENDPOINT_OUTPUT = OUTPUT_DIR / "e86_test_geometry_sequence_endpoint_light.npz"
OUTPUT = OUTPUT_DIR / "Round2_E86_Surface_Endpoint_V11_Blend_Test_Channel.npy"
REPORT = PATHS.analysis / "results" / "round2_surface_interaction_energy_model" / "e86_test_generate.json"

MP = 2
MH = 16
MV = 8
N_UE = 4
PATHS_PER_POINT = 128
TINY = 1e-30


def load_array(cache: Path, name: str, dtype=None) -> np.ndarray:
    value = np.load(cache / f"{name}.npy", mmap_mode="r")
    return np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)


def build_test_auxiliary() -> dict[str, np.ndarray]:
    setup, train_positions, test_positions, _, test_sites = load_positions()
    state = json.loads(SIONNA_STATE.read_text(encoding="utf-8"))
    tau = load_array(TEST_PATH_CACHE, "tau", np.float32)
    angles = load_array(TEST_PATH_CACHE, "angles", np.float32)
    coordinate = e84.sionna_coordinates(
        tau, angles, test_sites, state["configuration"], vertical_alignment=2.0
    )

    coefficient = load_array(TEST_PATH_CACHE, "coefficient")
    profile = np.abs(coefficient.reshape(-1, PATHS_PER_POINT, MP, MH, MV, N_UE)) ** 2
    port = profile.sum(axis=(3, 4), dtype=np.float64)
    port /= np.maximum(port.sum(axis=(2, 3), keepdims=True), TINY)

    center = np.concatenate(
        (train_positions[:, :2].mean(axis=0), np.asarray([1.5], dtype=np.float32))
    ).astype(np.float32)
    scale = np.concatenate(
        (
            np.maximum(train_positions[:, :2].std(axis=0), 1.0),
            np.asarray([20.0], dtype=np.float32),
        )
    ).astype(np.float32)
    vertices = load_array(TEST_PATH_CACHE, "vertices", np.float32)
    depth = load_array(TEST_PATH_CACHE, "interaction_count", np.int64)
    geometry = sequence_geometry_block(
        slice(0, len(test_positions)),
        vertices,
        depth,
        test_positions,
        test_sites,
        setup.bs_positions,
        center,
        scale,
    )
    sequence_source = {
        name: load_array(TEST_PATH_CACHE, name)
        for name in ("source", "interaction_count", "interactions", "objects", "primitives")
    }
    hashes = e84.sequence_hashes(sequence_source)
    buckets = np.stack(
        (
            hashes % np.uint64(Config().sequence_hash_buckets),
            (hashes >> np.uint64(32)) % np.uint64(Config().sequence_hash_buckets),
        ),
        axis=-1,
    ).astype(np.int64)
    return {
        "coordinate": coordinate.astype(np.float32),
        "port": port.astype(np.float32),
        "geometry": geometry.astype(np.float32),
        "hash": buckets,
        "depth": depth.astype(np.int64),
        "interactions": sequence_source["interactions"].astype(np.int64),
        "objects": sequence_source["objects"].astype(np.int64),
        "primitives": sequence_source["primitives"].astype(np.int64),
        "fraction": load_array(TEST_PATH_CACHE, "energy_fraction", np.float32),
        "presence": load_array(TEST_PATH_CACHE, "presence", np.float32),
    }


def test_batch(
    rows: np.ndarray,
    features: np.ndarray,
    auxiliary: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    fraction = auxiliary["fraction"][rows]
    presence = auxiliary["presence"][rows]
    anchored = np.maximum(fraction, 1e-12) * np.sqrt(
        np.maximum(presence, 1.0) / 24.0
    )
    port = auxiliary["port"][rows].reshape(len(rows), PATHS_PER_POINT, MP * N_UE)
    return {
        "base": torch.as_tensor(
            (features[rows].astype(np.float32) - mean) / std, device=device
        ),
        "base_log_mass": torch.as_tensor(
            np.log(np.maximum(anchored, 1e-12)), device=device
        ),
        "coordinate": torch.as_tensor(auxiliary["coordinate"][rows], device=device),
        "base_log_port": torch.as_tensor(
            np.log(np.maximum(port, 1e-8)), device=device
        ),
        "sequence_geometry": torch.as_tensor(
            auxiliary["geometry"][rows], device=device
        ),
        "sequence_hash": torch.as_tensor(auxiliary["hash"][rows], device=device),
        "depth": torch.as_tensor(auxiliary["depth"][rows], device=device),
        "interactions": torch.as_tensor(
            auxiliary["interactions"][rows], device=device
        ),
        "objects": torch.as_tensor(auxiliary["objects"][rows], device=device),
        "primitives": torch.as_tensor(
            auxiliary["primitives"][rows], device=device
        ),
    }


def circular_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    periods = np.asarray((MH, MV, 192.0), dtype=np.float32)
    phase = values * (2.0 * np.pi / periods)[None, None, None, :]
    weight = weights[..., None]
    sine = np.sum(weight * np.sin(phase), axis=0)
    cosine = np.sum(weight * np.cos(phase), axis=0)
    angle = np.mod(np.arctan2(sine, cosine), 2.0 * np.pi)
    return (angle * periods / (2.0 * np.pi)).astype(np.float32)


@torch.inference_mode()
def predict_ensemble(
    features: np.ndarray,
    auxiliary: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    rows = np.arange(len(features), dtype=np.int64)
    model_outputs = []
    checkpoint_paths = sorted(MODEL_DIR.glob("strict_fold*_geometry_sequence.pt"))
    if len(checkpoint_paths) != 5:
        raise FileNotFoundError("expected five strict geometry checkpoints")
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = Config(**checkpoint["config"])
        model = SurfaceInteractionModel(
            features.shape[-1], auxiliary["geometry"].shape[-1], config, "geometry"
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        outputs = []
        for start in range(0, len(rows), batch_size):
            local = rows[start : start + batch_size]
            outputs.append(model(test_batch(local, features, auxiliary, mean, std, device)))
        model_outputs.append(
            {
                key: np.concatenate(
                    [value[key].detach().cpu().numpy() for value in outputs], axis=0
                )
                for key in ("mass", "port", "coordinate")
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mass_stack = np.stack([value["mass"] for value in model_outputs], axis=0)
    port_stack = np.stack([value["port"] for value in model_outputs], axis=0)
    coordinate_stack = np.stack(
        [value["coordinate"] % np.asarray((MH, MV, 192.0), dtype=np.float32) for value in model_outputs],
        axis=0,
    )
    mass = np.mean(mass_stack, axis=0)
    mass /= np.maximum(mass.sum(axis=1, keepdims=True), TINY)
    port = np.mean(port_stack, axis=0)
    port /= np.maximum(port.sum(axis=(2, 3), keepdims=True), TINY)
    weights = mass_stack / np.maximum(mass_stack.sum(axis=0, keepdims=True), TINY)
    coordinate = circular_mean(coordinate_stack, weights)
    return {"mass": mass.astype(np.float32), "port": port.astype(np.float32), "coordinate": coordinate}


def main() -> None:
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    missing = [path for path in (TEST_PATH_CACHE, TEST_FEATURES, V11_TEST, SIONNA_STATE) if not path.exists()]
    if missing:
        raise FileNotFoundError([PATHS.relative(path) for path in missing])
    features = np.load(TEST_FEATURES, mmap_mode="r")
    auxiliary = build_test_auxiliary()
    prediction = predict_ensemble(features, auxiliary, device, batch_size=32)
    coefficient = np.sqrt(
        np.maximum(
            prediction["mass"][..., None]
            * prediction["port"].reshape(500, PATHS_PER_POINT, 8),
            0.0,
        )
    ).astype(np.complex64)
    _, endpoint_pas_flat, endpoint_pdp_flat = render_noncoherent(
        prediction["coordinate"], coefficient, device
    )
    endpoint_pas = torch.as_tensor(
        endpoint_pas_flat, device=device
    ).reshape(500, 2, 16, 8, 4, 192)
    endpoint_pdp = torch.as_tensor(endpoint_pdp_flat, device=device)
    np.savez_compressed(
        ENDPOINT_OUTPUT,
        mass=prediction["mass"],
        coordinate=prediction["coordinate"],
        port=prediction["port"],
        pas=endpoint_pas.cpu().numpy().astype(np.float16),
        pdp=endpoint_pdp.cpu().numpy().astype(np.float16),
    )
    np.savez_compressed(
        LIGHT_ENDPOINT_OUTPUT,
        mass=prediction["mass"],
        coordinate=prediction["coordinate"],
        port=prediction["port"],
    )

    base_np = np.load(V11_TEST, mmap_mode="r")
    base = torch.as_tensor(np.array(base_np, copy=True), device=device)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    desired_pas = (
        0.8 * torch.sqrt(base_pas.clamp_min(0.0))
        + 0.2 * torch.sqrt(endpoint_pas.clamp_min(0.0))
    ).square()
    desired_pas /= desired_pas.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)
    desired_pdp = (
        0.8 * torch.sqrt(base_pdp.clamp_min(0.0))
        + 0.2 * torch.sqrt(endpoint_pdp.clamp_min(0.0))
    ).square()
    desired_pdp /= desired_pdp.sum(dim=3, keepdim=True).clamp_min(TINY)
    output = project_joint(base, desired_pas, desired_pdp, 4, "pas_pdp")
    output_np = output.cpu().numpy().astype(np.complex64)
    np.save(OUTPUT, output_np)
    power_ratio = (
        output.abs().square().sum(dim=(1, 2, 3))
        / base.abs().square().sum(dim=(1, 2, 3)).clamp_min(TINY)
    ).cpu().numpy()
    payload = {
        "version": VERSION,
        "method": "E85 strict geometry-sequence endpoint + fixed sqrt PAS/PDP blend",
        "status": "candidate_generated_not_packaged",
        "policy": {
            "endpoint": "geometry_sequence",
            "model": "mean of five strict spatial-hole checkpoints",
            "pas_beta": 0.20,
            "pdp_beta": 0.20,
            "order": "pas_pdp",
            "cycles": 4,
            "absolute_sionna_phase_used": False,
            "v11_complex_carrier_preserved": True,
        },
        "output": {
            "path": PATHS.relative(OUTPUT),
            "shape": list(output_np.shape),
            "dtype": str(output_np.dtype),
            "finite": bool(np.isfinite(output_np.real).all() and np.isfinite(output_np.imag).all()),
            "bytes": int(OUTPUT.stat().st_size),
        },
        "endpoint": {
            "path": PATHS.relative(ENDPOINT_OUTPUT),
            "light_path": PATHS.relative(LIGHT_ENDPOINT_OUTPUT),
            "mass_shape": list(prediction["mass"].shape),
            "coordinate_shape": list(prediction["coordinate"].shape),
            "port_shape": list(prediction["port"].shape),
            "finite": bool(
                np.isfinite(prediction["mass"]).all()
                and np.isfinite(prediction["coordinate"]).all()
                and np.isfinite(prediction["port"]).all()
            ),
        },
        "power_ratio_mean": float(np.mean(power_ratio)),
        "power_ratio_max_abs_error": float(np.max(np.abs(power_ratio - 1.0))),
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
