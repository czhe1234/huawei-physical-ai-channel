#!/usr/bin/env python3
"""Train a surface-sequence path competition model on strict spatial folds."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np
import torch
from torch import nn

import analyze_path_candidate_information_audit as e84
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E85-surface-interaction-energy-model"
PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n4000_s24_k128_b096_d064_scatter1_sel106"
)
PATH_FEATURES = (
    PATHS.artifacts
    / "round2"
    / "explicit_path_pdp_predictor"
    / "path_features_train_n4000.npy"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
FULL_FOLD = (
    PATHS.artifacts / "round2" / "full_path_pdp_link_gate" / "train_fold_id.npy"
)
FULL_REGIME = (
    PATHS.artifacts
    / "round2"
    / "explicit_path_pdp_predictor"
    / "path_regime_train_n4000.npy"
)
SIONNA_STATE = (
    PATHS.artifacts
    / "round2"
    / "sionna_multiseed"
    / "primary_d16_base_v4lambda_1m"
    / "full"
    / "state.json"
)
ARTIFACT_DIR = PATHS.artifacts / "round2" / "surface_interaction_energy_model"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_surface_interaction_energy_model"
TARGET_PAS = ARTIFACT_DIR / "official_pas_cosine_sufficient_target_f32.npy"
TARGET_PDP = ARTIFACT_DIR / "official_pdp_cosine_sufficient_target_f32.npy"
POINT_POWER = ARTIFACT_DIR / "official_point_power_f64.npy"
BASE_COORDINATE = ARTIFACT_DIR / "sionna_aligned_coordinates_f32.npy"
PORT_PRIOR = ARTIFACT_DIR / "sionna_path_port_prior_f16.npy"
SEQUENCE_GEOMETRY = ARTIFACT_DIR / "sequence_geometry_f16.npy"
SEQUENCE_HASH = ARTIFACT_DIR / "sequence_hash_buckets_i32.npy"
PREPARE_REPORT = ANALYSIS_DIR / "surface_interaction_training_cache.json"
SCREEN_REPORT = ANALYSIS_DIR / "surface_interaction_screen.json"
FULL_REPORT = ANALYSIS_DIR / "surface_interaction_five_fold.json"
STRICT_REPORT = ANALYSIS_DIR / "surface_interaction_strict_holes.json"

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
PATHS_PER_POINT = 128
STEPS = 16
TINY = 1e-30
PRESENCE_SEED_COUNT = 24


@dataclass(frozen=True)
class Config:
    hidden: int = 96
    heads: int = 4
    sequence_layers: int = 2
    set_layers: int = 2
    feedforward: int = 256
    dropout: float = 0.08
    epochs: int = 48
    minimum_epochs: int = 16
    validation_interval: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 2e-4
    score_limit: float = 3.0
    port_limit: float = 2.0
    score_penalty: float = 0.002
    coordinate_penalty: float = 0.003
    port_penalty: float = 0.001
    random_state: int = 2026080795
    sequence_hash_buckets: int = 32749


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "screen", "full", "strict"),
        default="screen",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--outer", type=int, default=0)
    parser.add_argument("--validation-fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--sequence-layers", type=int, default=2)
    parser.add_argument("--set-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args()


def required_inputs() -> tuple:
    return (
        PATH_CACHE,
        PATH_FEATURES,
        OOF_META,
        FULL_FOLD,
        FULL_REGIME,
        SIONNA_STATE,
        PATHS.round2 / "Round2_Train_Channel.npy",
    )


def normalized_direction(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    norm = torch.sqrt(value.square().sum(dim=dims, keepdim=True, dtype=torch.float64))
    return torch.where(norm > 0.0, value / norm.clamp_min(TINY), torch.zeros_like(value))


def build_official_targets(device: torch.device, batch_size: int) -> dict[str, object]:
    train_h = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    count = len(train_h)
    pas_output = np.lib.format.open_memmap(
        TARGET_PAS,
        mode="w+",
        dtype=np.float32,
        shape=(count, MP, MH, MV, N_UE),
    )
    pdp_output = np.lib.format.open_memmap(
        TARGET_PDP,
        mode="w+",
        dtype=np.float32,
        shape=(count, MP, N_UE, S_SC),
    )
    power_output = np.lib.format.open_memmap(
        POINT_POWER, mode="w+", dtype=np.float64, shape=(count,)
    )
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        channel = torch.as_tensor(
            np.array(train_h[start:end], copy=True), device=device
        )
        shaped = channel.reshape(-1, MP, MH, MV, N_UE, S_SC)
        pas_power = torch.fft.fft2(shaped, dim=(2, 3)).abs().square().float()
        pas_direction = normalized_direction(pas_power, (1, 2, 3)).mean(dim=5)
        delay_power = torch.fft.ifft(shaped, dim=5).abs().square().float()
        delay_direction = normalized_direction(delay_power, (5,)).mean(dim=(2, 3))
        pas_output[start:end] = pas_direction.cpu().numpy().astype(np.float32)
        pdp_output[start:end] = delay_direction.cpu().numpy().astype(np.float32)
        power_output[start:end] = (
            channel.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64).cpu().numpy()
        )
        print(f"official target cache {end}/{count}", flush=True)
    pas_output.flush()
    pdp_output.flush()
    power_output.flush()
    return {
        "pas_shape": list(pas_output.shape),
        "pdp_shape": list(pdp_output.shape),
        "nonzero_points": int(np.sum(np.asarray(power_output) > 0.0)),
    }


def sequence_geometry_block(
    rows: slice,
    vertices: np.ndarray,
    depth: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    local_vertices = np.array(vertices[rows], dtype=np.float32, copy=True)
    local_depth = np.asarray(depth[rows], dtype=np.int64)
    point = positions[rows, None, None, :].astype(np.float32)
    bs = bs_positions[sites[rows]][:, None, None, :].astype(np.float32)
    step = np.arange(STEPS, dtype=np.int64)[None, None, :]
    actual = step < local_depth[..., None]
    token_mask = step < np.maximum(local_depth, 1)[..., None]

    midpoint = 0.5 * (point + bs)
    local_vertices = np.where(
        ((local_depth == 0)[..., None, None] & (step == 0)[..., None]),
        midpoint,
        local_vertices,
    )
    previous = np.empty_like(local_vertices)
    previous[..., 0, :] = bs[..., 0, :]
    previous[..., 1:, :] = local_vertices[..., :-1, :]
    following = np.empty_like(local_vertices)
    following[..., :-1, :] = local_vertices[..., 1:, :]
    following[..., -1, :] = point[..., 0, :]
    last = step == np.maximum(local_depth - 1, 0)[..., None]
    following = np.where(last[..., None], point, following)

    incoming = local_vertices - previous
    outgoing = following - local_vertices
    incoming_length = np.linalg.norm(incoming, axis=-1, keepdims=True)
    outgoing_length = np.linalg.norm(outgoing, axis=-1, keepdims=True)
    incoming_unit = incoming / np.maximum(incoming_length, 1e-6)
    outgoing_unit = outgoing / np.maximum(outgoing_length, 1e-6)
    turn = np.sum(incoming_unit * outgoing_unit, axis=-1, keepdims=True)
    normalized_vertex = (local_vertices - center) / scale
    relative_ue = (local_vertices - point) / scale
    relative_bs = (local_vertices - bs) / scale
    step_fraction = np.broadcast_to(step / float(STEPS - 1), actual.shape)[..., None]
    depth_fraction = np.broadcast_to(
        local_depth[..., None, None] / float(STEPS),
        (*local_depth.shape, STEPS, 1),
    )
    output = np.concatenate(
        (
            normalized_vertex,
            relative_ue,
            relative_bs,
            incoming_unit,
            np.log1p(incoming_length) / 5.0,
            outgoing_unit,
            np.log1p(outgoing_length) / 5.0,
            turn,
            step_fraction.astype(np.float32),
            depth_fraction.astype(np.float32),
            actual[..., None].astype(np.float32),
        ),
        axis=-1,
    ).astype(np.float32)
    output *= token_mask[..., None]
    return output


def build_path_auxiliary_cache(batch_size: int) -> dict[str, object]:
    setup, positions, _, sites, _ = load_positions()
    count = len(positions)
    state = json.loads(SIONNA_STATE.read_text(encoding="utf-8"))
    tau = np.load(PATH_CACHE / "tau.npy", mmap_mode="r")
    angles = np.load(PATH_CACHE / "angles.npy", mmap_mode="r")
    coordinate = e84.sionna_coordinates(
        tau, angles, sites, state["configuration"], vertical_alignment=2.0
    )
    np.save(BASE_COORDINATE, coordinate.astype(np.float32))

    coefficient = np.load(PATH_CACHE / "coefficient.npy", mmap_mode="r")
    port_output = np.lib.format.open_memmap(
        PORT_PRIOR,
        mode="w+",
        dtype=np.float16,
        shape=(count, PATHS_PER_POINT, MP, N_UE),
    )
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        shaped = np.asarray(coefficient[start:end]).reshape(
            end - start, PATHS_PER_POINT, MP, MH, MV, N_UE
        )
        power = np.abs(shaped) ** 2
        profile = power.sum(axis=(3, 4), dtype=np.float64)
        profile /= np.maximum(profile.sum(axis=(2, 3), keepdims=True), TINY)
        port_output[start:end] = profile.astype(np.float16)
        print(f"port prior cache {end}/{count}", flush=True)
    port_output.flush()

    vertices = np.load(PATH_CACHE / "vertices.npy", mmap_mode="r")
    depth = np.load(PATH_CACHE / "interaction_count.npy", mmap_mode="r")
    center = np.concatenate(
        (positions[:, :2].mean(axis=0), np.asarray([1.5], dtype=np.float32))
    ).astype(np.float32)
    scale = np.concatenate(
        (
            np.maximum(positions[:, :2].std(axis=0), 1.0),
            np.asarray([20.0], dtype=np.float32),
        )
    ).astype(np.float32)
    probe = sequence_geometry_block(
        slice(0, 1),
        vertices,
        depth,
        positions,
        sites,
        setup.bs_positions,
        center,
        scale,
    )
    geometry_output = np.lib.format.open_memmap(
        SEQUENCE_GEOMETRY,
        mode="w+",
        dtype=np.float16,
        shape=(count, PATHS_PER_POINT, STEPS, probe.shape[-1]),
    )
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        geometry_output[start:end] = sequence_geometry_block(
            slice(start, end),
            vertices,
            depth,
            positions,
            sites,
            setup.bs_positions,
            center,
            scale,
        ).astype(np.float16)
        print(f"sequence geometry cache {end}/{count}", flush=True)
    geometry_output.flush()

    sequence_source = {
        name: np.load(PATH_CACHE / f"{name}.npy", mmap_mode="r")
        for name in (
            "source",
            "interaction_count",
            "interactions",
            "objects",
            "primitives",
        )
    }
    hashes = e84.sequence_hashes(sequence_source)
    buckets = np.stack(
        (
            hashes % np.uint64(Config().sequence_hash_buckets),
            (hashes >> np.uint64(32)) % np.uint64(Config().sequence_hash_buckets),
        ),
        axis=-1,
    ).astype(np.int32)
    np.save(SEQUENCE_HASH, buckets)
    return {
        "coordinate_shape": list(coordinate.shape),
        "port_shape": list(port_output.shape),
        "sequence_geometry_shape": list(geometry_output.shape),
        "sequence_hash_shape": list(buckets.shape),
        "geometry_center": center.tolist(),
        "geometry_scale": scale.tolist(),
    }


def prepare_caches(device: torch.device, batch_size: int, force: bool) -> dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    target_ready = all(path.is_file() for path in (TARGET_PAS, TARGET_PDP, POINT_POWER))
    path_ready = all(
        path.is_file()
        for path in (BASE_COORDINATE, PORT_PRIOR, SEQUENCE_GEOMETRY, SEQUENCE_HASH)
    )
    started = time.time()
    target = (
        build_official_targets(device, batch_size)
        if force or not target_ready
        else {
            "pas_shape": list(np.load(TARGET_PAS, mmap_mode="r").shape),
            "pdp_shape": list(np.load(TARGET_PDP, mmap_mode="r").shape),
            "nonzero_points": int(np.sum(np.load(POINT_POWER, mmap_mode="r") > 0.0)),
        }
    )
    path = (
        build_path_auxiliary_cache(batch_size)
        if force or not path_ready
        else {
            "coordinate_shape": list(np.load(BASE_COORDINATE, mmap_mode="r").shape),
            "port_shape": list(np.load(PORT_PRIOR, mmap_mode="r").shape),
            "sequence_geometry_shape": list(
                np.load(SEQUENCE_GEOMETRY, mmap_mode="r").shape
            ),
            "sequence_hash_shape": list(np.load(SEQUENCE_HASH, mmap_mode="r").shape),
        }
    )
    report = {
        "experiment": VERSION,
        "target": target,
        "path": path,
        "seconds": float(time.time() - started),
    }
    PREPARE_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


class SurfaceInteractionModel(nn.Module):
    def __init__(
        self,
        base_dim: int,
        sequence_dim: int,
        config: Config,
        sequence_mode: str,
    ) -> None:
        super().__init__()
        if sequence_mode not in ("none", "geometry", "surface"):
            raise ValueError(f"unsupported sequence mode: {sequence_mode}")
        self.sequence_mode = sequence_mode
        self.config = config
        hidden = config.hidden
        self.base_input = nn.Sequential(
            nn.Linear(base_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.sequence_input = nn.Linear(sequence_dim, hidden)
        self.interaction_embedding = nn.Embedding(8, hidden)
        self.object_embedding = nn.Embedding(16, hidden)
        self.primitive_embedding = nn.Embedding(24038, hidden)
        self.step_embedding = nn.Embedding(STEPS, hidden)
        self.hash_embedding_a = nn.Embedding(config.sequence_hash_buckets, hidden)
        self.hash_embedding_b = nn.Embedding(config.sequence_hash_buckets, hidden)
        sequence_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            sequence_layer, num_layers=config.sequence_layers
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        set_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            set_layer, num_layers=config.set_layers
        )
        self.score_head = nn.Linear(hidden, 1)
        self.coordinate_head = nn.Linear(hidden, 3)
        self.port_head = nn.Linear(hidden, MP * N_UE)
        for layer in (self.score_head, self.coordinate_head, self.port_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def sequence_state(
        self,
        continuous: torch.Tensor,
        interaction: torch.Tensor,
        objects: torch.Tensor,
        primitives: torch.Tensor,
        depth: torch.Tensor,
        sequence_hash: torch.Tensor,
    ) -> torch.Tensor:
        batch, paths, steps, _ = continuous.shape
        if self.sequence_mode == "none":
            return torch.zeros(
                batch, paths, self.config.hidden, device=continuous.device
            )
        flat = batch * paths
        step_ids = torch.arange(steps, device=continuous.device)
        hidden = self.sequence_input(continuous)
        hidden = (
            hidden
            + self.interaction_embedding(interaction.clamp(0, 7))
            + self.step_embedding(step_ids)[None, None]
        )
        if self.sequence_mode == "surface":
            hidden = (
                hidden
                + self.object_embedding(objects.clamp(0, 15))
                + self.primitive_embedding(primitives.clamp(0, 24037))
            )
        hidden = hidden.reshape(flat, steps, self.config.hidden)
        token_mask = step_ids[None] < depth.reshape(flat, 1).clamp_min(1)
        hidden = self.sequence_encoder(hidden, src_key_padding_mask=~token_mask)
        weight = token_mask.to(hidden.dtype)
        pooled = (hidden * weight[..., None]).sum(dim=1) / weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        pooled = pooled.reshape(batch, paths, self.config.hidden)
        if self.sequence_mode == "surface":
            pooled = (
                pooled
                + self.hash_embedding_a(sequence_hash[..., 0])
                + self.hash_embedding_b(sequence_hash[..., 1])
            )
        return pooled

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
        hidden = self.fusion(torch.cat((base, sequence), dim=-1))
        hidden = self.set_encoder(hidden)
        score_residual = self.config.score_limit * torch.tanh(
            self.score_head(hidden)[..., 0]
        )
        coordinate_residual = torch.tanh(self.coordinate_head(hidden)) * torch.tensor(
            (2.0, 4.0, 2.0), device=hidden.device
        )
        port_residual = self.config.port_limit * torch.tanh(self.port_head(hidden))
        mass = torch.softmax(batch["base_log_mass"] + score_residual, dim=1)
        port = torch.softmax(batch["base_log_port"] + port_residual, dim=2).reshape(
            len(hidden), PATHS_PER_POINT, MP, N_UE
        )
        coordinate = batch["coordinate"] + coordinate_residual
        return {
            "mass": mass,
            "port": port,
            "coordinate": coordinate,
            "score_residual": score_residual,
            "coordinate_residual": coordinate_residual,
            "port_residual": port_residual,
        }


def spectral_kernels(coordinate: torch.Tensor) -> tuple[torch.Tensor, ...]:
    h_index = torch.arange(MH, dtype=torch.float32, device=coordinate.device)
    v_index = torch.arange(MV, dtype=torch.float32, device=coordinate.device)
    d_index = torch.arange(S_SC, dtype=torch.float32, device=coordinate.device)
    h_wave = torch.exp(
        2j * torch.pi * coordinate[..., 0, None] * h_index / float(MH)
    )
    v_wave = torch.exp(
        2j * torch.pi * coordinate[..., 1, None] * v_index / float(MV)
    )
    d_wave = torch.exp(
        -2j * torch.pi * coordinate[..., 2, None] * d_index / float(S_SC)
    )
    horizontal = torch.fft.fft(h_wave, dim=-1).abs().square().float()
    vertical = torch.fft.fft(v_wave, dim=-1).abs().square().float()
    delay = torch.fft.ifft(d_wave, dim=-1).abs().square().float()
    horizontal /= horizontal.sum(dim=-1, keepdim=True).clamp_min(TINY)
    vertical /= vertical.sum(dim=-1, keepdim=True).clamp_min(TINY)
    delay /= delay.sum(dim=-1, keepdim=True).clamp_min(TINY)
    return horizontal, vertical, delay


def render_spectra(output: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    horizontal, vertical, delay = spectral_kernels(output["coordinate"])
    energy = output["mass"][..., None, None] * output["port"]
    pas = torch.einsum("bkpu,bkh,bkv->bphvu", energy, horizontal, vertical)
    pas /= pas.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)
    pdp = torch.einsum("bkpu,bkd->bpud", energy, delay)
    pdp /= pdp.sum(dim=3, keepdim=True).clamp_min(TINY)
    return pas, pdp


def cosine_sufficient_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    norm_dims: tuple[int, ...],
    average_dims: tuple[int, ...],
) -> torch.Tensor:
    norm = torch.sqrt(prediction.square().sum(dim=norm_dims, dtype=torch.float64))
    dot = (prediction * target).sum(dim=norm_dims, dtype=torch.float64)
    value = dot / torch.where(norm > 0.0, norm, torch.ones_like(norm))
    return value.mean(dim=average_dims, dtype=torch.float64)


def objective(
    output: dict[str, torch.Tensor],
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pas, pdp = render_spectra(output)
    pas_score = cosine_sufficient_score(pas, target_pas, (1, 2, 3), (1,))
    pdp_score = cosine_sufficient_score(pdp, target_pdp, (3,), (1, 2))
    spectral_score = 0.5 * (pas_score + pdp_score)
    score_penalty = output["score_residual"].square().mean(dim=1)
    coordinate_scale = torch.tensor((2.0, 4.0, 2.0), device=pas.device)
    coordinate_penalty = (
        output["coordinate_residual"] / coordinate_scale
    ).square().mean(dim=(1, 2))
    port_penalty = output["port_residual"].square().mean(dim=(1, 2))
    loss = (
        1.0
        - spectral_score
        + config.score_penalty * score_penalty
        + config.coordinate_penalty * coordinate_penalty
        + config.port_penalty * port_penalty
    ).mean()
    return loss, {
        "pas": pas_score,
        "pdp": pdp_score,
        "score": spectral_score,
    }


class DataStore:
    def __init__(self) -> None:
        self.base = np.load(PATH_FEATURES, mmap_mode="r")
        self.fraction = np.load(PATH_CACHE / "energy_fraction.npy", mmap_mode="r")
        self.presence = np.load(PATH_CACHE / "presence.npy", mmap_mode="r")
        self.coordinate = np.load(BASE_COORDINATE, mmap_mode="r")
        self.port = np.load(PORT_PRIOR, mmap_mode="r")
        self.sequence_geometry = np.load(SEQUENCE_GEOMETRY, mmap_mode="r")
        self.sequence_hash = np.load(SEQUENCE_HASH, mmap_mode="r")
        self.depth = np.load(PATH_CACHE / "interaction_count.npy", mmap_mode="r")
        self.interactions = np.load(PATH_CACHE / "interactions.npy", mmap_mode="r")
        self.objects = np.load(PATH_CACHE / "objects.npy", mmap_mode="r")
        self.primitives = np.load(PATH_CACHE / "primitives.npy", mmap_mode="r")
        self.target_pas = np.load(TARGET_PAS, mmap_mode="r")
        self.target_pdp = np.load(TARGET_PDP, mmap_mode="r")
        self.power = np.load(POINT_POWER, mmap_mode="r")
        self.fold = np.load(FULL_FOLD).astype(np.int8)
        self.regime = np.load(FULL_REGIME).astype(np.int8)
        _, _, _, self.site, _ = load_positions()
        metadata = np.load(OOF_META, allow_pickle=False)
        self.strict_ids = metadata["query_ids"].astype(np.int64)
        self.strict_fold = metadata["fold_id"].astype(np.int8)

    def normalization(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        total = np.zeros(self.base.shape[-1], dtype=np.float64)
        square = np.zeros_like(total)
        count = 0
        for start in range(0, len(rows), 128):
            values = np.asarray(self.base[rows[start : start + 128]], dtype=np.float64)
            total += values.sum(axis=(0, 1))
            square += np.square(values).sum(axis=(0, 1))
            count += values.shape[0] * values.shape[1]
        mean = total / max(count, 1)
        variance = np.maximum(square / max(count, 1) - mean * mean, 1e-8)
        std = np.sqrt(variance)
        std[std < 1e-4] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    def batch(
        self,
        rows: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        device: torch.device,
        sequence_rows: np.ndarray | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if sequence_rows is None:
            sequence_rows = rows
        fraction = np.asarray(self.fraction[rows], dtype=np.float32)
        presence = np.asarray(self.presence[rows], dtype=np.float32)
        anchored = np.maximum(fraction, 1e-12) * np.sqrt(
            np.maximum(presence, 1.0) / float(PRESENCE_SEED_COUNT)
        )
        port = np.asarray(self.port[rows], dtype=np.float32).reshape(
            len(rows), PATHS_PER_POINT, MP * N_UE
        )
        batch = {
            "base": torch.as_tensor(
                (np.asarray(self.base[rows], dtype=np.float32) - mean) / std,
                device=device,
            ),
            "base_log_mass": torch.as_tensor(
                np.log(np.maximum(anchored, 1e-12)), device=device
            ),
            "coordinate": torch.as_tensor(
                np.asarray(self.coordinate[rows], dtype=np.float32), device=device
            ),
            "base_log_port": torch.as_tensor(
                np.log(np.maximum(port, 1e-8)), device=device
            ),
            "sequence_geometry": torch.as_tensor(
                np.asarray(self.sequence_geometry[sequence_rows], dtype=np.float32),
                device=device,
            ),
            "sequence_hash": torch.as_tensor(
                np.asarray(self.sequence_hash[sequence_rows], dtype=np.int64),
                device=device,
            ),
            "depth": torch.as_tensor(
                np.asarray(self.depth[sequence_rows], dtype=np.int64), device=device
            ),
            "interactions": torch.as_tensor(
                np.asarray(self.interactions[sequence_rows], dtype=np.int64),
                device=device,
            ),
            "objects": torch.as_tensor(
                np.asarray(self.objects[sequence_rows], dtype=np.int64), device=device
            ),
            "primitives": torch.as_tensor(
                np.asarray(self.primitives[sequence_rows], dtype=np.int64), device=device
            ),
        }
        target_pas = torch.as_tensor(
            np.asarray(self.target_pas[rows], dtype=np.float32), device=device
        )
        target_pdp = torch.as_tensor(
            np.asarray(self.target_pdp[rows], dtype=np.float32), device=device
        )
        return batch, target_pas, target_pdp


def raw_output(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mass = torch.softmax(batch["base_log_mass"], dim=1)
    port = torch.softmax(batch["base_log_port"], dim=2).reshape(
        len(mass), PATHS_PER_POINT, MP, N_UE
    )
    zeros_score = torch.zeros_like(mass)
    zeros_coordinate = torch.zeros_like(batch["coordinate"])
    zeros_port = torch.zeros_like(batch["base_log_port"])
    return {
        "mass": mass,
        "port": port,
        "coordinate": batch["coordinate"],
        "score_residual": zeros_score,
        "coordinate_residual": zeros_coordinate,
        "port_residual": zeros_port,
    }


@torch.inference_mode()
def evaluate_rows(
    model: SurfaceInteractionModel | None,
    store: DataStore,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
    sequence_rows: np.ndarray | None = None,
) -> dict[str, object]:
    if model is not None:
        model.eval()
    pas_values = []
    pdp_values = []
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        local_sequence = (
            None if sequence_rows is None else sequence_rows[start : start + batch_size]
        )
        batch, target_pas, target_pdp = store.batch(
            local, mean, std, device, local_sequence
        )
        output = raw_output(batch) if model is None else model(batch)
        _, values = objective(output, target_pas, target_pdp, Config())
        pas_values.append(values["pas"].cpu().numpy())
        pdp_values.append(values["pdp"].cpu().numpy())
    pas = np.concatenate(pas_values)
    pdp = np.concatenate(pdp_values)
    return {
        "count": int(len(rows)),
        "pas": float(np.mean(pas)),
        "pdp": float(np.mean(pdp)),
        "pas_pdp_mean": float(np.mean(0.5 * (pas + pdp))),
        "per_point_pas": pas,
        "per_point_pdp": pdp,
    }


@torch.inference_mode()
def predict_rows(
    model: SurfaceInteractionModel,
    store: DataStore,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
    sequence_rows: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    mass = np.zeros((len(rows), PATHS_PER_POINT), dtype=np.float32)
    coordinate = np.zeros((len(rows), PATHS_PER_POINT, 3), dtype=np.float32)
    port = np.zeros(
        (len(rows), PATHS_PER_POINT, MP, N_UE), dtype=np.float32
    )
    pas = np.zeros(len(rows), dtype=np.float64)
    pdp = np.zeros(len(rows), dtype=np.float64)
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        local = rows[start:end]
        local_sequence = (
            None if sequence_rows is None else sequence_rows[start:end]
        )
        batch, target_pas, target_pdp = store.batch(
            local, mean, std, device, local_sequence
        )
        output = model(batch)
        _, values = objective(output, target_pas, target_pdp, model.config)
        mass[start:end] = output["mass"].cpu().numpy().astype(np.float32)
        coordinate[start:end] = (
            output["coordinate"].cpu().numpy()
            % np.asarray((MH, MV, S_SC), dtype=np.float32)
        ).astype(np.float32)
        port[start:end] = output["port"].cpu().numpy().astype(np.float32)
        pas[start:end] = values["pas"].cpu().numpy()
        pdp[start:end] = values["pdp"].cpu().numpy()
    return {
        "mass": mass,
        "coordinate": coordinate,
        "port": port,
        "pas": pas,
        "pdp": pdp,
    }


def train_model(
    store: DataStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray | None,
    mean: np.ndarray,
    std: np.ndarray,
    config: Config,
    device: torch.device,
    batch_size: int,
    epochs: int,
    seed: int,
    sequence_mode: str,
) -> tuple[SurfaceInteractionModel, list[dict[str, float]], int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SurfaceInteractionModel(
        store.base.shape[-1],
        store.sequence_geometry.shape[-1],
        config,
        sequence_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=config.learning_rate * 0.08
    )
    rng = np.random.default_rng(seed)
    train_rows = fit_rows[np.asarray(store.power[fit_rows]) > 0.0]
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = config.minimum_epochs
    best_score = -math.inf
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_rows)
        losses = []
        scores = []
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            batch, target_pas, target_pdp = store.batch(rows, mean, std, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, values = objective(output, target_pas, target_pdp, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            scores.append(float(values["score"].mean().detach().item()))
        scheduler.step()
        record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "train_score": float(np.mean(scores)),
        }
        if (
            validation_rows is not None
            and epoch >= config.minimum_epochs
            and epoch % config.validation_interval == 0
        ):
            validation = evaluate_rows(
                model,
                store,
                validation_rows,
                mean,
                std,
                device,
                batch_size,
            )
            record["validation_score"] = float(validation["pas_pdp_mean"])
            if record["validation_score"] > best_score:
                best_score = record["validation_score"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(record)
        if epoch % config.validation_interval == 0 or epoch == epochs:
            message = (
                f"sequence={sequence_mode} epoch={epoch}/{epochs} "
                f"train={record['train_score']:.6f}"
            )
            if "validation_score" in record:
                message += f" validation={record['validation_score']:.6f}"
            print(message, flush=True)
    if validation_rows is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def shuffled_sequence_rows(store: DataStore, rows: np.ndarray, seed: int) -> np.ndarray:
    output = np.array(rows, copy=True)
    rng = np.random.default_rng(seed)
    for site in np.unique(store.site[rows]):
        for regime in np.unique(store.regime[rows]):
            local = np.flatnonzero(
                (store.site[rows] == site) & (store.regime[rows] == regime)
            )
            if len(local) <= 1:
                continue
            output[local] = rows[rng.permutation(local)]
    return output


def clean_metrics(value: dict[str, object]) -> dict[str, object]:
    return {
        key: item
        for key, item in value.items()
        if not key.startswith("per_point_")
    }


def screen(args: argparse.Namespace, config: Config, device: torch.device) -> None:
    if args.outer == args.validation_fold:
        raise ValueError("outer and validation fold must differ")
    store = DataStore()
    outer_rows = np.flatnonzero(store.fold == args.outer).astype(np.int64)
    validation_rows = np.flatnonzero(store.fold == args.validation_fold).astype(np.int64)
    screen_fit = np.flatnonzero(
        (store.fold != args.outer) & (store.fold != args.validation_fold)
    ).astype(np.int64)
    refit_rows = np.flatnonzero(store.fold != args.outer).astype(np.int64)
    strict_outer = store.strict_ids[store.fold[store.strict_ids] == args.outer]
    mean_screen, std_screen = store.normalization(screen_fit)
    mean_refit, std_refit = store.normalization(refit_rows)
    raw_full = evaluate_rows(
        None, store, outer_rows, mean_refit, std_refit, device, args.batch_size
    )
    raw_strict = evaluate_rows(
        None, store, strict_outer, mean_refit, std_refit, device, args.batch_size
    )
    results: dict[str, object] = {}
    prediction_payload: dict[str, np.ndarray] = {
        "outer_rows": outer_rows.astype(np.int32),
        "strict_outer_rows": strict_outer.astype(np.int32),
    }
    training = []
    for branch_index, (name, sequence_mode) in enumerate(
        (
            ("no_sequence", "none"),
            ("geometry_sequence", "geometry"),
            ("surface_sequence", "surface"),
        )
    ):
        _, screen_history, best_epoch = train_model(
            store,
            screen_fit,
            validation_rows,
            mean_screen,
            std_screen,
            config,
            device,
            args.batch_size,
            config.epochs,
            config.random_state + 1000 * branch_index,
            sequence_mode,
        )
        model, refit_history, _ = train_model(
            store,
            refit_rows,
            None,
            mean_refit,
            std_refit,
            config,
            device,
            args.batch_size,
            best_epoch,
            config.random_state + 5000 + 1000 * branch_index,
            sequence_mode,
        )
        full = evaluate_rows(
            model, store, outer_rows, mean_refit, std_refit, device, args.batch_size
        )
        strict = evaluate_rows(
            model, store, strict_outer, mean_refit, std_refit, device, args.batch_size
        )
        row: dict[str, object] = {
            "best_epoch": best_epoch,
            "full_outer": clean_metrics(full),
            "strict_outer": clean_metrics(strict),
            "full_delta_vs_raw": float(
                full["pas_pdp_mean"] - raw_full["pas_pdp_mean"]
            ),
            "strict_delta_vs_raw": float(
                strict["pas_pdp_mean"] - raw_strict["pas_pdp_mean"]
            ),
        }
        prediction_payload[f"{name}_full_pas"] = full["per_point_pas"]
        prediction_payload[f"{name}_full_pdp"] = full["per_point_pdp"]
        prediction_payload[f"{name}_strict_pas"] = strict["per_point_pas"]
        prediction_payload[f"{name}_strict_pdp"] = strict["per_point_pdp"]
        if sequence_mode != "none":
            shuffle_full_rows = shuffled_sequence_rows(
                store, outer_rows, config.random_state + 9000
            )
            shuffle_strict_rows = shuffled_sequence_rows(
                store, strict_outer, config.random_state + 10000
            )
            shuffled_full = evaluate_rows(
                model,
                store,
                outer_rows,
                mean_refit,
                std_refit,
                device,
                args.batch_size,
                shuffle_full_rows,
            )
            shuffled_strict = evaluate_rows(
                model,
                store,
                strict_outer,
                mean_refit,
                std_refit,
                device,
                args.batch_size,
                shuffle_strict_rows,
            )
            row["sequence_shuffle_full"] = clean_metrics(shuffled_full)
            row["sequence_shuffle_strict"] = clean_metrics(shuffled_strict)
            row["full_margin_over_shuffle"] = float(
                full["pas_pdp_mean"] - shuffled_full["pas_pdp_mean"]
            )
            row["strict_margin_over_shuffle"] = float(
                strict["pas_pdp_mean"] - shuffled_strict["pas_pdp_mean"]
            )
        results[name] = row
        checkpoint = ARTIFACT_DIR / f"screen_outer{args.outer}_{name}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(config),
                "feature_mean": mean_refit,
                "feature_std": std_refit,
                "best_epoch": best_epoch,
                "sequence_mode": sequence_mode,
            },
            checkpoint,
        )
        training.append(
            {
                "branch": name,
                "screen_history": screen_history,
                "refit_history": refit_history,
                "checkpoint": PATHS.relative(checkpoint),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    sequence = results["geometry_sequence"]
    no_sequence = results["no_sequence"]
    decision = {
        "minimum_full_gain_over_raw": 0.005,
        "minimum_strict_gain_over_raw": 0.003,
        "minimum_full_sequence_margin": 0.0005,
        "full_gain_over_raw": sequence["full_delta_vs_raw"],
        "strict_gain_over_raw": sequence["strict_delta_vs_raw"],
        "full_margin_over_no_sequence": float(
            sequence["full_outer"]["pas_pdp_mean"]
            - no_sequence["full_outer"]["pas_pdp_mean"]
        ),
        "strict_margin_over_no_sequence": float(
            sequence["strict_outer"]["pas_pdp_mean"]
            - no_sequence["strict_outer"]["pas_pdp_mean"]
        ),
        "full_margin_over_shuffle": sequence["full_margin_over_shuffle"],
        "strict_margin_over_shuffle": sequence["strict_margin_over_shuffle"],
    }
    decision["geometry_sequence_passed"] = bool(
        decision["full_gain_over_raw"] >= decision["minimum_full_gain_over_raw"]
        and decision["strict_gain_over_raw"] >= decision["minimum_strict_gain_over_raw"]
        and decision["full_margin_over_no_sequence"]
        >= decision["minimum_full_sequence_margin"]
        and decision["strict_margin_over_no_sequence"] >= 0.0
        and decision["full_margin_over_shuffle"]
        >= decision["minimum_full_sequence_margin"]
    )
    decision["aggregate_model_scale_up_passed"] = bool(
        no_sequence["full_delta_vs_raw"] >= decision["minimum_full_gain_over_raw"]
        and no_sequence["strict_delta_vs_raw"]
        >= decision["minimum_strict_gain_over_raw"]
    )
    decision["passed"] = decision["aggregate_model_scale_up_passed"]
    prediction_path = ARTIFACT_DIR / f"screen_outer{args.outer}_point_metrics.npz"
    np.savez_compressed(prediction_path, **prediction_payload)
    report = {
        "experiment": VERSION,
        "status": "screen_complete",
        "primary_hypothesis": (
            "Ordered surface interactions explain official path survival, angular correction, "
            "and energy better than aggregate path features alone."
        ),
        "configuration": asdict(config),
        "outer": args.outer,
        "validation_fold": args.validation_fold,
        "screen_fit_count": int(len(screen_fit)),
        "refit_count": int(len(refit_rows)),
        "outer_count": int(len(outer_rows)),
        "strict_outer_count": int(len(strict_outer)),
        "strict_fold_mapping": (
            "The strict 500 points are assigned by FULL_FOLD for this model; the older "
            "OOF_META fold labels define a different experiment and are not used here."
        ),
        "raw": {
            "full_outer": clean_metrics(raw_full),
            "strict_outer": clean_metrics(raw_strict),
        },
        "results": results,
        "decision": decision,
        "training": training,
        "prediction_cache": PATHS.relative(prediction_path),
    }
    SCREEN_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "raw": report["raw"],
                "results": results,
                "decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def summarize_prediction(
    pas: np.ndarray,
    pdp: np.ndarray,
    rows: np.ndarray,
    fold: np.ndarray,
) -> dict[str, object]:
    return {
        "count": int(len(rows)),
        "pas": float(np.mean(pas[rows])),
        "pdp": float(np.mean(pdp[rows])),
        "pas_pdp_mean": float(np.mean(0.5 * (pas[rows] + pdp[rows]))),
        "fold": [
            {
                "fold": outer,
                "count": int(np.sum(fold[rows] == outer)),
                "pas_pdp_mean": float(
                    np.mean(0.5 * (pas[rows][fold[rows] == outer] + pdp[rows][fold[rows] == outer]))
                ),
            }
            for outer in range(5)
        ],
    }


def full_five_fold(args: argparse.Namespace, config: Config, device: torch.device) -> None:
    store = DataStore()
    count = len(store.base)
    branches = (("no_sequence", "none"), ("geometry_sequence", "geometry"))
    raw = evaluate_rows(
        None,
        store,
        np.arange(count, dtype=np.int64),
        np.zeros(store.base.shape[-1], dtype=np.float32),
        np.ones(store.base.shape[-1], dtype=np.float32),
        device,
        args.batch_size,
    )
    raw_pas = raw["per_point_pas"]
    raw_pdp = raw["per_point_pdp"]
    branch_results: dict[str, object] = {}
    training_rows: list[dict[str, object]] = []
    for branch_index, (name, sequence_mode) in enumerate(branches):
        oof_mass = np.zeros((count, PATHS_PER_POINT), dtype=np.float32)
        oof_coordinate = np.zeros((count, PATHS_PER_POINT, 3), dtype=np.float32)
        oof_port = np.zeros(
            (count, PATHS_PER_POINT, MP, N_UE), dtype=np.float32
        )
        oof_pas = np.zeros(count, dtype=np.float64)
        oof_pdp = np.zeros(count, dtype=np.float64)
        oof_shuffle_pas = np.zeros(count, dtype=np.float64)
        oof_shuffle_pdp = np.zeros(count, dtype=np.float64)
        for outer in range(5):
            validation_fold = (outer + 1) % 5
            held_rows = np.flatnonzero(store.fold == outer).astype(np.int64)
            validation_rows = np.flatnonzero(
                store.fold == validation_fold
            ).astype(np.int64)
            screen_fit = np.flatnonzero(
                (store.fold != outer) & (store.fold != validation_fold)
            ).astype(np.int64)
            refit_rows = np.flatnonzero(store.fold != outer).astype(np.int64)
            mean_screen, std_screen = store.normalization(screen_fit)
            mean_refit, std_refit = store.normalization(refit_rows)
            _, screen_history, best_epoch = train_model(
                store,
                screen_fit,
                validation_rows,
                mean_screen,
                std_screen,
                config,
                device,
                args.batch_size,
                config.epochs,
                config.random_state + 10000 * branch_index + 1000 * outer,
                sequence_mode,
            )
            model, refit_history, _ = train_model(
                store,
                refit_rows,
                None,
                mean_refit,
                std_refit,
                config,
                device,
                args.batch_size,
                best_epoch,
                config.random_state + 50000 + 10000 * branch_index + 1000 * outer,
                sequence_mode,
            )
            held = predict_rows(
                model,
                store,
                held_rows,
                mean_refit,
                std_refit,
                device,
                args.batch_size,
            )
            oof_mass[held_rows] = held["mass"]
            oof_coordinate[held_rows] = held["coordinate"]
            oof_port[held_rows] = held["port"]
            oof_pas[held_rows] = held["pas"]
            oof_pdp[held_rows] = held["pdp"]
            if sequence_mode != "none":
                shuffled_rows = shuffled_sequence_rows(
                    store, held_rows, config.random_state + 90000 + outer
                )
                shuffled = predict_rows(
                    model,
                    store,
                    held_rows,
                    mean_refit,
                    std_refit,
                    device,
                    args.batch_size,
                    shuffled_rows,
                )
                oof_shuffle_pas[held_rows] = shuffled["pas"]
                oof_shuffle_pdp[held_rows] = shuffled["pdp"]
            checkpoint = ARTIFACT_DIR / f"fold{outer}_{name}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(config),
                    "feature_mean": mean_refit,
                    "feature_std": std_refit,
                    "best_epoch": best_epoch,
                    "sequence_mode": sequence_mode,
                    "outer": outer,
                },
                checkpoint,
            )
            training_rows.append(
                {
                    "branch": name,
                    "outer": outer,
                    "validation_fold": validation_fold,
                    "best_epoch": best_epoch,
                    "screen_history": screen_history,
                    "refit_history": refit_history,
                    "checkpoint": PATHS.relative(checkpoint),
                }
            )
            print(
                f"full branch={name} outer={outer} epoch={best_epoch} "
                f"score={np.mean(0.5 * (held['pas'] + held['pdp'])):.6f}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        endpoint = ARTIFACT_DIR / f"oof_{name}_path_endpoint.npz"
        np.savez_compressed(
            endpoint,
            mass=oof_mass,
            coordinate=oof_coordinate,
            port=oof_port,
            pas=oof_pas,
            pdp=oof_pdp,
            fold=store.fold,
        )
        all_rows = np.arange(count, dtype=np.int64)
        strict_rows = store.strict_ids
        result = {
            "all_4000": summarize_prediction(
                oof_pas, oof_pdp, all_rows, store.fold
            ),
            "strict_500": summarize_prediction(
                oof_pas, oof_pdp, strict_rows, store.fold
            ),
            "all_delta_vs_raw": float(
                np.mean(0.5 * (oof_pas + oof_pdp))
                - np.mean(0.5 * (raw_pas + raw_pdp))
            ),
            "strict_delta_vs_raw": float(
                np.mean(0.5 * (oof_pas[strict_rows] + oof_pdp[strict_rows]))
                - np.mean(0.5 * (raw_pas[strict_rows] + raw_pdp[strict_rows]))
            ),
            "endpoint": PATHS.relative(endpoint),
        }
        if sequence_mode != "none":
            result["shuffle_all_4000"] = summarize_prediction(
                oof_shuffle_pas, oof_shuffle_pdp, all_rows, store.fold
            )
            result["shuffle_strict_500"] = summarize_prediction(
                oof_shuffle_pas, oof_shuffle_pdp, strict_rows, store.fold
            )
            result["all_margin_over_shuffle"] = float(
                np.mean(0.5 * (oof_pas + oof_pdp))
                - np.mean(0.5 * (oof_shuffle_pas + oof_shuffle_pdp))
            )
            result["strict_margin_over_shuffle"] = float(
                np.mean(0.5 * (oof_pas[strict_rows] + oof_pdp[strict_rows]))
                - np.mean(
                    0.5
                    * (
                        oof_shuffle_pas[strict_rows]
                        + oof_shuffle_pdp[strict_rows]
                    )
                )
            )
        branch_results[name] = result

    raw_all = summarize_prediction(
        raw_pas,
        raw_pdp,
        np.arange(count, dtype=np.int64),
        store.fold,
    )
    raw_strict = summarize_prediction(
        raw_pas, raw_pdp, store.strict_ids, store.fold
    )
    selected_name = max(
        branch_results,
        key=lambda name: branch_results[name]["strict_500"]["pas_pdp_mean"],
    )
    selected = branch_results[selected_name]
    fold_delta = [
        row["pas_pdp_mean"] - raw_strict["fold"][index]["pas_pdp_mean"]
        for index, row in enumerate(selected["strict_500"]["fold"])
    ]
    decision = {
        "selected_branch": selected_name,
        "minimum_strict_gain": 0.01,
        "minimum_positive_strict_folds": 4,
        "strict_gain": selected["strict_delta_vs_raw"],
        "strict_fold_delta": fold_delta,
        "strict_positive_folds": int(np.sum(np.asarray(fold_delta) > 0.0)),
    }
    decision["passed_path_model_gate"] = bool(
        decision["strict_gain"] >= decision["minimum_strict_gain"]
        and decision["strict_positive_folds"]
        >= decision["minimum_positive_strict_folds"]
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "validation": (
            "Every point is predicted by the model whose FULL_FOLD outer excludes it; "
            "OOF_META fold labels are not mixed with FULL_FOLD."
        ),
        "configuration": asdict(config),
        "raw": {"all_4000": raw_all, "strict_500": raw_strict},
        "branches": branch_results,
        "decision": decision,
        "training": training_rows,
    }
    FULL_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "raw": report["raw"],
                "branches": branch_results,
                "decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def strict_spatial_holes(
    args: argparse.Namespace, config: Config, device: torch.device
) -> None:
    store = DataStore()
    branches = (("no_sequence", "none"), ("geometry_sequence", "geometry"))
    query_ids = store.strict_ids
    fold_id = store.strict_fold
    global_to_row = np.full(len(store.base), -1, dtype=np.int64)
    global_to_row[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    raw = evaluate_rows(
        None,
        store,
        query_ids,
        np.zeros(store.base.shape[-1], dtype=np.float32),
        np.ones(store.base.shape[-1], dtype=np.float32),
        device,
        args.batch_size,
    )
    raw_pas = raw["per_point_pas"]
    raw_pdp = raw["per_point_pdp"]
    all_points = np.arange(len(store.base), dtype=np.int64)
    branch_results: dict[str, object] = {}
    training_rows: list[dict[str, object]] = []
    for branch_index, (name, sequence_mode) in enumerate(branches):
        oof_mass = np.zeros((len(query_ids), PATHS_PER_POINT), dtype=np.float32)
        oof_coordinate = np.zeros(
            (len(query_ids), PATHS_PER_POINT, 3), dtype=np.float32
        )
        oof_port = np.zeros(
            (len(query_ids), PATHS_PER_POINT, MP, N_UE), dtype=np.float32
        )
        oof_pas = np.zeros(len(query_ids), dtype=np.float64)
        oof_pdp = np.zeros(len(query_ids), dtype=np.float64)
        oof_shuffle_pas = np.zeros(len(query_ids), dtype=np.float64)
        oof_shuffle_pdp = np.zeros(len(query_ids), dtype=np.float64)
        for outer in range(5):
            validation_fold = (outer + 1) % 5
            held = query_ids[fold_id == outer]
            validation = query_ids[fold_id == validation_fold]
            held_mask = np.zeros(len(store.base), dtype=bool)
            held_mask[held] = True
            validation_mask = np.zeros(len(store.base), dtype=bool)
            validation_mask[validation] = True
            screen_fit = all_points[~held_mask & ~validation_mask]
            refit = all_points[~held_mask]
            mean_screen, std_screen = store.normalization(screen_fit)
            mean_refit, std_refit = store.normalization(refit)
            _, screen_history, best_epoch = train_model(
                store,
                screen_fit,
                validation,
                mean_screen,
                std_screen,
                config,
                device,
                args.batch_size,
                config.epochs,
                config.random_state + 200000 + 10000 * branch_index + 1000 * outer,
                sequence_mode,
            )
            model, refit_history, _ = train_model(
                store,
                refit,
                None,
                mean_refit,
                std_refit,
                config,
                device,
                args.batch_size,
                best_epoch,
                config.random_state + 250000 + 10000 * branch_index + 1000 * outer,
                sequence_mode,
            )
            prediction = predict_rows(
                model,
                store,
                held,
                mean_refit,
                std_refit,
                device,
                args.batch_size,
            )
            rows = global_to_row[held]
            if np.any(rows < 0):
                raise RuntimeError("strict held rows are not in OOF metadata")
            oof_mass[rows] = prediction["mass"]
            oof_coordinate[rows] = prediction["coordinate"]
            oof_port[rows] = prediction["port"]
            oof_pas[rows] = prediction["pas"]
            oof_pdp[rows] = prediction["pdp"]
            if sequence_mode != "none":
                shuffled_held = shuffled_sequence_rows(
                    store, held, config.random_state + 290000 + outer
                )
                shuffled = predict_rows(
                    model,
                    store,
                    held,
                    mean_refit,
                    std_refit,
                    device,
                    args.batch_size,
                    shuffled_held,
                )
                oof_shuffle_pas[rows] = shuffled["pas"]
                oof_shuffle_pdp[rows] = shuffled["pdp"]
            checkpoint = ARTIFACT_DIR / f"strict_fold{outer}_{name}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(config),
                    "feature_mean": mean_refit,
                    "feature_std": std_refit,
                    "best_epoch": best_epoch,
                    "sequence_mode": sequence_mode,
                    "outer": outer,
                    "validation_fold": validation_fold,
                },
                checkpoint,
            )
            training_rows.append(
                {
                    "branch": name,
                    "outer": outer,
                    "validation_fold": validation_fold,
                    "screen_fit_count": int(len(screen_fit)),
                    "refit_count": int(len(refit)),
                    "held_count": int(len(held)),
                    "best_epoch": best_epoch,
                    "screen_history": screen_history,
                    "refit_history": refit_history,
                    "checkpoint": PATHS.relative(checkpoint),
                }
            )
            print(
                f"strict branch={name} outer={outer} epoch={best_epoch} "
                f"score={np.mean(0.5 * (prediction['pas'] + prediction['pdp'])):.6f}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        endpoint = ARTIFACT_DIR / f"strict_oof_{name}_path_endpoint.npz"
        np.savez_compressed(
            endpoint,
            query_ids=query_ids.astype(np.int32),
            fold_id=fold_id,
            mass=oof_mass,
            coordinate=oof_coordinate,
            port=oof_port,
            pas=oof_pas,
            pdp=oof_pdp,
        )
        rows = np.arange(len(query_ids), dtype=np.int64)
        result = {
            "strict_500": summarize_prediction(oof_pas, oof_pdp, rows, fold_id),
            "delta_vs_raw": float(
                np.mean(0.5 * (oof_pas + oof_pdp))
                - np.mean(0.5 * (raw_pas + raw_pdp))
            ),
            "endpoint": PATHS.relative(endpoint),
        }
        if sequence_mode != "none":
            result["shuffle_strict_500"] = summarize_prediction(
                oof_shuffle_pas, oof_shuffle_pdp, rows, fold_id
            )
            result["margin_over_shuffle"] = float(
                np.mean(0.5 * (oof_pas + oof_pdp))
                - np.mean(0.5 * (oof_shuffle_pas + oof_shuffle_pdp))
            )
        branch_results[name] = result

    rows = np.arange(len(query_ids), dtype=np.int64)
    raw_summary = summarize_prediction(raw_pas, raw_pdp, rows, fold_id)
    selected_name = max(
        branch_results,
        key=lambda name: branch_results[name]["strict_500"]["pas_pdp_mean"],
    )
    selected = branch_results[selected_name]
    fold_delta = [
        selected["strict_500"]["fold"][outer]["pas_pdp_mean"]
        - raw_summary["fold"][outer]["pas_pdp_mean"]
        for outer in range(5)
    ]
    decision = {
        "selected_branch": selected_name,
        "minimum_gain": 0.01,
        "minimum_positive_folds": 4,
        "gain": selected["delta_vs_raw"],
        "fold_delta": fold_delta,
        "positive_folds": int(np.sum(np.asarray(fold_delta) > 0.0)),
        "minimum_score_to_beat_e83_local": 0.603200,
        "score": selected["strict_500"]["pas_pdp_mean"],
    }
    decision["passed_path_model_gate"] = bool(
        decision["gain"] >= decision["minimum_gain"]
        and decision["positive_folds"] >= decision["minimum_positive_folds"]
        and decision["score"] >= decision["minimum_score_to_beat_e83_local"]
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "validation": (
            "Canonical fixed 500-point contiguous spatial holes. Each outer model excludes "
            "all held-hole labels; another complete hole selects the epoch."
        ),
        "configuration": asdict(config),
        "raw": raw_summary,
        "branches": branch_results,
        "decision": decision,
        "training": training_rows,
    }
    STRICT_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "raw": raw_summary,
                "branches": branch_results,
                "decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    missing = [PATHS.relative(path) for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    if not 0 <= args.outer < 5 or not 0 <= args.validation_fold < 5:
        raise ValueError("fold ids must be in [0, 4]")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    config = Config(
        hidden=args.hidden,
        sequence_layers=args.sequence_layers,
        set_layers=args.set_layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    prepare = prepare_caches(device, args.batch_size, args.force_prepare)
    print(json.dumps({"prepare": prepare}, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "screen":
        screen(args, config, device)
    elif args.stage == "full":
        full_five_fold(args, config, device)
    elif args.stage == "strict":
        strict_spatial_holes(args, config, device)


if __name__ == "__main__":
    main()
