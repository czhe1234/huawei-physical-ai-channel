#!/usr/bin/env python3
"""Infer a shared radio surface field and render official spectral corrections."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
import trimesh

import train_e141_base_conditioned_joint_spectral_operator as e141
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import FOLD_CACHE, OOF_META, statistics
from run_v3_anchored_neural_rbf import (
    Config as ContextConfig,
    ContextTable,
    build_train_contexts,
    set_seed,
    validation_context,
)


VERSION = "Round2-E145-surface-field-inverse-renderer"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "surface_field_inverse_renderer"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_surface_field_inverse_renderer"
SURFACE_CACHE = ARTIFACT_DIR / "official_surface_sequences.npz"
REPORT = ANALYSIS_DIR / "surface_field_inverse_renderer.json"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
MAP_PATH = PATHS.round2 / "Round2_Map.ply"

RAY_OFFSETS_M = (-4.0, -2.0, 0.0, 2.0, 4.0)
MAX_HITS = 16
TOKEN_DIM = 18
TINY = 1e-30
V7_SCORE = 0.644456346
V12_SCORE = 0.677520807


@dataclass(frozen=True)
class Config:
    pas_rank: int = 96
    pdp_rank: int = 96
    patch_dim: int = 24
    token_hidden: int = 192
    query_hidden: int = 320
    heads: int = 6
    surface_layers: int = 2
    fusion_layers: int = 3
    dropout: float = 0.08
    epochs: int = 36
    batch_size: int = 56
    learning_rate: float = 3e-4
    weight_decay: float = 2e-4
    coefficient_weight: float = 0.08
    action_regularization: float = 0.002
    output_bound: float = 3.0
    minimum_object_support: int = 4
    minimum_facet_support: int = 4
    beta: float = 0.20
    cycles: int = 4
    order: str = "pas_pdp"
    required_screen_gain: float = 0.002
    required_full_gain: float = 0.010
    seed: int = 20_260_809_145


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default="3")
    parser.add_argument("--modes", default="actual,control")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--pas-rank", type=int, default=Config.pas_rank)
    parser.add_argument("--pdp-rank", type=int, default=Config.pdp_rank)
    parser.add_argument("--force-surface-cache", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def face_component_labels(face_count: int, adjacency: np.ndarray) -> np.ndarray:
    parent = np.arange(face_count, dtype=np.int32)
    rank = np.zeros(face_count, dtype=np.int8)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    for left, right in np.asarray(adjacency, dtype=np.int64):
        root_left = find(int(left))
        root_right = find(int(right))
        if root_left == root_right:
            continue
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1
    roots = np.asarray([find(index) for index in range(face_count)], dtype=np.int32)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32) + 1


def canonical_normals(value: np.ndarray) -> np.ndarray:
    normal = np.asarray(value, dtype=np.float64).copy()
    flip = (normal[:, 0] < -1e-8) | (
        (np.abs(normal[:, 0]) <= 1e-8) & (normal[:, 1] < -1e-8)
    )
    flip |= (
        (np.abs(normal[:, 0]) <= 1e-8)
        & (np.abs(normal[:, 1]) <= 1e-8)
        & (normal[:, 2] < 0.0)
    )
    normal[flip] *= -1.0
    return normal


def build_control_sources(
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    hit_count = mask.sum(axis=2)
    total_hits = hit_count.sum(axis=1)
    center_hits = hit_count[:, len(RAY_OFFSETS_M) // 2]
    clear_rays = (hit_count == 0).sum(axis=1)
    distance = np.linalg.norm(positions - bs_positions[site], axis=1)
    source = np.arange(len(positions), dtype=np.int32)
    for site_value in np.unique(site):
        rows = np.flatnonzero(site == site_value)
        raw = np.column_stack(
            (
                distance[rows] / 20.0,
                total_hits[rows] / 5.0,
                center_hits[rows] / 2.0,
                clear_rays[rows] / 2.0,
            )
        )
        count = min(48, len(rows))
        _, neighbors = cKDTree(raw).query(raw, k=count, workers=-1)
        if count == 1:
            neighbors = neighbors[:, None]
        for local, row in enumerate(rows):
            candidates = rows[np.asarray(neighbors[local], dtype=np.int64)]
            candidates = candidates[candidates != row]
            if len(candidates) == 0:
                continue
            separation = np.linalg.norm(
                positions[candidates, :2] - positions[row, None, :2], axis=1
            )
            source[row] = int(candidates[np.argmax(separation)])
    selected = source.astype(np.int64)
    return source, {
        "self_fraction": float(np.mean(selected == np.arange(len(source)))),
        "median_spatial_separation_m": float(
            np.median(np.linalg.norm(positions[selected, :2] - positions[:, :2], axis=1))
        ),
        "median_bs_distance_difference_m": float(
            np.median(np.abs(distance[selected] - distance))
        ),
        "median_total_hit_difference": float(
            np.median(np.abs(total_hits[selected] - total_hits))
        ),
        "median_center_hit_difference": float(
            np.median(np.abs(center_hits[selected] - center_hits))
        ),
    }


def build_surface_cache(force: bool) -> dict[str, object]:
    if SURFACE_CACHE.is_file() and not force:
        cached = np.load(SURFACE_CACHE, allow_pickle=False)
        return {
            "path": PATHS.relative(SURFACE_CACHE),
            "reused": True,
            "point_count": int(len(cached["token"])),
            "object_count": int(cached["object_count"]),
            "facet_count": int(cached["facet_count"]),
            "valid_hit_count": int(cached["mask"].sum()),
            "control": json.loads(str(cached["control_json"])),
        }

    setup, train_positions, _, train_site, _ = load_positions()
    mesh = trimesh.load(MAP_PATH, process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh, got {type(mesh).__name__}")
    face_count = len(mesh.faces)
    object_of_face = face_component_labels(face_count, mesh.face_adjacency)
    facet_of_face = np.zeros(face_count, dtype=np.int32)
    for facet_id, faces in enumerate(mesh.facets, start=1):
        facet_of_face[np.asarray(faces, dtype=np.int64)] = facet_id
    face_normal = canonical_normals(np.asarray(mesh.face_normals, dtype=np.float64))
    face_area = np.asarray(mesh.area_faces, dtype=np.float64)

    positions = np.asarray(train_positions, dtype=np.float64)
    serving = setup.bs_positions[train_site]
    delta_xy = positions[:, :2] - serving[:, :2]
    norm_xy = np.maximum(np.linalg.norm(delta_xy, axis=1, keepdims=True), 1e-9)
    perpendicular = np.column_stack((-delta_xy[:, 1], delta_xy[:, 0])) / norm_xy
    offsets = np.asarray(RAY_OFFSETS_M, dtype=np.float64)
    targets = np.repeat(positions[:, None, :], len(offsets), axis=1)
    targets[:, :, :2] += perpendicular[:, None, :] * offsets[None, :, None]
    origins = np.repeat(serving[:, None, :], len(offsets), axis=1)
    flat_origins = origins.reshape(-1, 3)
    flat_targets = targets.reshape(-1, 3)
    vectors = flat_targets - flat_origins
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors / np.maximum(lengths[:, None], 1e-12)

    point_count = len(positions)
    ray_count = len(offsets)
    token = np.zeros((point_count, ray_count, MAX_HITS, TOKEN_DIM), dtype=np.float32)
    object_id = np.zeros((point_count, ray_count, MAX_HITS), dtype=np.int32)
    facet_id = np.zeros_like(object_id)
    mask = np.zeros_like(object_id, dtype=bool)
    raw_counts = np.zeros(point_count * ray_count, dtype=np.int16)
    retained_counts = np.zeros_like(raw_counts)

    ray_batch = 2048
    for start in range(0, len(flat_origins), ray_batch):
        stop = min(start + ray_batch, len(flat_origins))
        locations, local_ray, triangle = mesh.ray.intersects_location(
            flat_origins[start:stop],
            directions[start:stop],
            multiple_hits=True,
        )
        if len(local_ray) == 0:
            continue
        global_ray = np.asarray(local_ray, dtype=np.int64) + start
        triangle = np.asarray(triangle, dtype=np.int64)
        distance = np.linalg.norm(locations - flat_origins[global_ray], axis=1)
        keep = (distance > 0.50) & (distance < lengths[global_ray] - 0.75)
        locations = np.asarray(locations[keep], dtype=np.float64)
        global_ray = global_ray[keep]
        triangle = triangle[keep]
        distance = distance[keep]
        order = np.lexsort((distance, global_ray))
        locations = locations[order]
        global_ray = global_ray[order]
        triangle = triangle[order]
        distance = distance[order]
        for ray in np.unique(global_ray):
            rows = np.flatnonzero(global_ray == ray)
            raw_counts[ray] = len(rows)
            if len(rows) > 1:
                local_distance = distance[rows]
                deduplicate = np.concatenate(
                    ([True], np.diff(local_distance) > 0.35)
                )
                rows = rows[deduplicate]
            rows = rows[:MAX_HITS]
            retained_counts[ray] = len(rows)
            point = int(ray // ray_count)
            fan = int(ray % ray_count)
            ray_length = max(float(lengths[ray]), 1e-9)
            normal = face_normal[triangle[rows]]
            incidence = np.abs(normal @ directions[ray])
            local = np.zeros((len(rows), TOKEN_DIM), dtype=np.float32)
            local[:, 0:3] = locations[rows].astype(np.float32)
            local[:, 3:6] = normal.astype(np.float32)
            local[:, 6] = (distance[rows] / ray_length).astype(np.float32)
            local[:, 7] = (1.0 - distance[rows] / ray_length).astype(np.float32)
            local[:, 8] = incidence.astype(np.float32)
            local[:, 9] = np.log1p(face_area[triangle[rows]]).astype(np.float32)
            local[:, 10] = np.float32(offsets[fan] / 4.0)
            local[:, 11] = np.float32(directions[ray, 0])
            local[:, 12] = np.float32(directions[ray, 1])
            local[:, 13] = np.float32(directions[ray, 2])
            local[:, 14] = np.float32(ray_length / 250.0)
            local[:, 15] = (
                np.arange(len(rows), dtype=np.float32) / max(MAX_HITS - 1, 1)
            )
            local[:, 16] = np.float32(len(rows) / MAX_HITS)
            local[:, 17] = np.abs(normal[:, 2]).astype(np.float32)
            token[point, fan, : len(rows)] = local
            object_id[point, fan, : len(rows)] = object_of_face[triangle[rows]]
            facet_id[point, fan, : len(rows)] = facet_of_face[triangle[rows]]
            mask[point, fan, : len(rows)] = True
        print(f"E145 surface rays {stop}/{len(flat_origins)}", flush=True)

    control_source, control = build_control_sources(
        positions, train_site, setup.bs_positions, mask
    )
    if not np.isfinite(token).all():
        raise ValueError("non-finite E145 surface token")
    np.savez_compressed(
        SURFACE_CACHE,
        token=token.astype(np.float16),
        object_id=object_id,
        facet_id=facet_id,
        mask=mask,
        control_source=control_source,
        object_count=np.int32(object_of_face.max(initial=0)),
        facet_count=np.int32(facet_of_face.max(initial=0)),
        raw_counts=raw_counts.reshape(point_count, ray_count),
        retained_counts=retained_counts.reshape(point_count, ray_count),
        control_json=np.asarray(json.dumps(control, sort_keys=True)),
    )
    return {
        "path": PATHS.relative(SURFACE_CACHE),
        "reused": False,
        "point_count": point_count,
        "object_count": int(object_of_face.max(initial=0)),
        "facet_count": int(facet_of_face.max(initial=0)),
        "valid_hit_count": int(mask.sum()),
        "raw_hit_count": int(raw_counts.sum()),
        "truncated_ray_fraction": float(np.mean(raw_counts > MAX_HITS)),
        "control": control,
    }


def compact_identifiers(
    values: np.ndarray,
    mask: np.ndarray,
    fit_sources: np.ndarray,
    minimum_support: int,
) -> tuple[np.ndarray, int, dict[str, float | int]]:
    maximum = int(values.max(initial=0))
    support = np.zeros(maximum + 1, dtype=np.int32)
    for row in np.unique(fit_sources):
        local = np.unique(values[int(row)][mask[int(row)]])
        local = local[local > 0]
        support[local] += 1
    keep = np.flatnonzero(support >= minimum_support)
    mapping = np.zeros(maximum + 1, dtype=np.int32)
    mapping[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    remapped = mapping[values]
    valid = mask & (values > 0)
    covered = valid & (remapped > 0)
    return remapped, len(keep), {
        "raw_id_count": int(np.sum(support > 0)),
        "retained_id_count": int(len(keep)),
        "token_coverage": float(np.sum(covered) / max(np.sum(valid), 1)),
    }


class SurfaceInverseRenderer(nn.Module):
    def __init__(
        self,
        query_dim: int,
        object_count: int,
        facet_count: int,
        config: Config,
    ) -> None:
        super().__init__()
        self.object_embedding = nn.Embedding(
            object_count + 1, config.patch_dim, padding_idx=0
        )
        self.facet_embedding = nn.Embedding(
            facet_count + 1, config.patch_dim, padding_idx=0
        )
        self.ray_embedding = nn.Embedding(len(RAY_OFFSETS_M), config.patch_dim)
        token_input = TOKEN_DIM + 3 * config.patch_dim
        self.token_encoder = nn.Sequential(
            nn.Linear(token_input, config.token_hidden),
            nn.GELU(),
            nn.LayerNorm(config.token_hidden),
            nn.Linear(config.token_hidden, config.token_hidden),
        )
        self.query_token = nn.Sequential(
            nn.Linear(query_dim, config.token_hidden),
            nn.GELU(),
            nn.LayerNorm(config.token_hidden),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.token_hidden,
            nhead=config.heads,
            dim_feedforward=2 * config.token_hidden,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.surface_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.surface_layers
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, config.query_hidden),
            nn.GELU(),
            nn.LayerNorm(config.query_hidden),
            nn.Linear(config.query_hidden, config.query_hidden),
        )
        self.surface_film = nn.Sequential(
            nn.Linear(config.token_hidden, 2 * config.query_hidden),
            nn.Tanh(),
        )
        self.fusion = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(config.query_hidden),
                nn.Linear(config.query_hidden, 3 * config.query_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(3 * config.query_hidden, config.query_hidden),
            )
            for _ in range(config.fusion_layers)
        )
        output = config.pas_rank + config.pdp_rank
        self.head = nn.Sequential(
            nn.LayerNorm(config.query_hidden),
            nn.Linear(config.query_hidden, config.query_hidden),
            nn.GELU(),
            nn.Linear(config.query_hidden, output),
        )
        nn.init.normal_(self.object_embedding.weight, std=0.02)
        nn.init.normal_(self.facet_embedding.weight, std=0.02)
        with torch.no_grad():
            self.object_embedding.weight[0].zero_()
            self.facet_embedding.weight[0].zero_()
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.output_bound = float(config.output_bound)

    def forward(
        self,
        query: torch.Tensor,
        token: torch.Tensor,
        object_id: torch.Tensor,
        facet_id: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, rays, hits, _ = token.shape
        ray_id = torch.arange(rays, device=token.device)[None, :, None]
        ray = self.ray_embedding(ray_id).expand(batch, rays, hits, -1)
        encoded = self.token_encoder(
            torch.cat(
                (
                    token,
                    self.object_embedding(object_id),
                    self.facet_embedding(facet_id),
                    ray,
                ),
                dim=3,
            )
        ).reshape(batch, rays * hits, -1)
        flat_mask = mask.reshape(batch, rays * hits)
        query_token = self.query_token(query)[:, None, :]
        sequence = torch.cat((query_token, encoded), dim=1)
        padding = torch.cat(
            (
                torch.zeros((batch, 1), dtype=torch.bool, device=mask.device),
                ~flat_mask,
            ),
            dim=1,
        )
        surface = self.surface_encoder(sequence, src_key_padding_mask=padding)[:, 0]
        hidden = self.query_encoder(query)
        scale, shift = self.surface_film(surface).chunk(2, dim=1)
        hidden = hidden * (1.0 + 0.35 * scale) + 0.35 * shift
        for block in self.fusion:
            hidden = hidden + block(hidden)
        raw = self.head(hidden)
        return self.output_bound * torch.tanh(raw / self.output_bound)


@dataclass
class FoldData:
    state: torch.Tensor
    position: torch.Tensor
    site: torch.Tensor
    pas_amplitude: torch.Tensor
    pdp_amplitude: torch.Tensor
    donor_code: torch.Tensor
    pas_components: torch.Tensor
    pdp_components: torch.Tensor
    pas_scale: torch.Tensor
    pdp_scale: torch.Tensor
    token: torch.Tensor
    object_id: torch.Tensor
    facet_id: torch.Tensor
    mask: torch.Tensor
    source: torch.Tensor


def query_inputs(
    context: dict[str, torch.Tensor],
    rows: torch.Tensor,
    guard: torch.Tensor | None,
    data: FoldData,
    pas_mean: torch.Tensor,
    pas_input_scale: torch.Tensor,
    pdp_mean: torch.Tensor,
    pdp_input_scale: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    query_ids = context["query_ids"][rows]
    if context["donor_ids"].ndim == 3:
        if guard is None:
            raise ValueError("training contexts require guard indices")
        donor_ids = context["donor_ids"][rows, guard]
        weights = context["weights"][rows, guard]
        distances = context["distances"][rows, guard]
    else:
        donor_ids = context["donor_ids"][rows]
        weights = context["weights"][rows]
        distances = context["distances"][rows]
    pas_anchor = e141.anchor_torch(
        data.pas_amplitude,
        donor_ids,
        weights,
        (e141.MP, e141.MH, e141.MV, e141.N_UE),
        (1, 2, 3),
    )
    pdp_anchor = e141.anchor_torch(
        data.pdp_amplitude,
        donor_ids,
        weights,
        (e141.MP, e141.N_UE, e141.S_SC),
        (3,),
    )
    anchor_code = torch.cat(
        (
            e141.encode_anchor_torch(
                pas_anchor, data.pas_components, pas_mean, pas_input_scale
            ),
            e141.encode_anchor_torch(
                pdp_anchor, data.pdp_components, pdp_mean, pdp_input_scale
            ),
        ),
        dim=1,
    )
    donor_code = data.donor_code[donor_ids]
    mean_code = donor_code.mean(dim=1)
    weighted_code = torch.sum(weights[:, :, None] * donor_code, dim=1)
    spread = torch.sqrt(
        (donor_code - mean_code[:, None, :]).square().mean(dim=1).clamp_min(1e-8)
    )
    summary = torch.stack(
        (
            distances[:, 0] / 20.0,
            distances.mean(dim=1) / 20.0,
            distances.std(dim=1) / 20.0,
            weights.abs().sum(dim=1),
        ),
        dim=1,
    )
    query = torch.cat(
        (data.state[query_ids], anchor_code, weighted_code, spread, summary), dim=1
    )
    return query, query_ids, pas_anchor, pdp_anchor, donor_ids, weights


def forward_batch(
    model: SurfaceInverseRenderer,
    context: dict[str, torch.Tensor],
    rows: torch.Tensor,
    guard: torch.Tensor | None,
    data: FoldData,
    pas_mean: torch.Tensor,
    pas_input_scale: torch.Tensor,
    pdp_mean: torch.Tensor,
    pdp_input_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query, query_ids, pas_anchor, pdp_anchor, _, _ = query_inputs(
        context,
        rows,
        guard,
        data,
        pas_mean,
        pas_input_scale,
        pdp_mean,
        pdp_input_scale,
    )
    surface_rows = data.source[query_ids]
    coefficient = model(
        query,
        data.token[surface_rows],
        data.object_id[surface_rows],
        data.facet_id[surface_rows],
        data.mask[surface_rows],
    )
    pas_count = data.pas_components.shape[0]
    pas_delta = (
        coefficient[:, :pas_count] * data.pas_scale
    ) @ data.pas_components
    pdp_delta = (
        coefficient[:, pas_count:] * data.pdp_scale
    ) @ data.pdp_components
    pas = e141.normalize_amplitude_torch(
        pas_anchor + pas_delta,
        (e141.MP, e141.MH, e141.MV, e141.N_UE),
        (1, 2, 3),
    )
    pdp = e141.normalize_amplitude_torch(
        pdp_anchor + pdp_delta,
        (e141.MP, e141.N_UE, e141.S_SC),
        (3,),
    )
    return pas, pdp, coefficient, query_ids, torch.cat((pas_anchor, pdp_anchor), dim=1)


@torch.inference_mode()
def predict(
    model: SurfaceInverseRenderer,
    table: ContextTable,
    data: FoldData,
    pas_basis: e141.ResidualBasis,
    pdp_basis: e141.ResidualBasis,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    context = e141.tensor_context(table, device)
    pas_mean = torch.as_tensor(pas_basis.input_mean, device=device)
    pas_input_scale = torch.as_tensor(pas_basis.input_scale, device=device)
    pdp_mean = torch.as_tensor(pdp_basis.input_mean, device=device)
    pdp_input_scale = torch.as_tensor(pdp_basis.input_scale, device=device)
    rows_out: dict[str, list[np.ndarray]] = {
        "pas": [],
        "pdp": [],
        "anchor_pas": [],
        "anchor_pdp": [],
        "coefficient": [],
    }
    for start in range(0, len(table.query_ids), batch_size):
        stop = min(start + batch_size, len(table.query_ids))
        rows = torch.arange(start, stop, device=device)
        pas, pdp, coefficient, _, anchor = forward_batch(
            model,
            context,
            rows,
            None,
            data,
            pas_mean,
            pas_input_scale,
            pdp_mean,
            pdp_input_scale,
        )
        rows_out["pas"].append(pas.cpu().numpy())
        rows_out["pdp"].append(pdp.cpu().numpy())
        rows_out["anchor_pas"].append(anchor[:, : e141.PAS_SIZE].cpu().numpy())
        rows_out["anchor_pdp"].append(anchor[:, e141.PAS_SIZE :].cpu().numpy())
        rows_out["coefficient"].append(coefficient.cpu().numpy())
    return {
        key: np.concatenate(value).astype(np.float32) for key, value in rows_out.items()
    }


def fold_result_paths(outer: int, mode: str) -> tuple[Path, Path, Path]:
    stem = f"fold{outer}_{mode}"
    return (
        ANALYSIS_DIR / f"{stem}.json",
        ARTIFACT_DIR / f"{stem}_metrics.npz",
        ARTIFACT_DIR / f"{stem}.pt",
    )


def train_mode(
    outer: int,
    mode: str,
    config: Config,
    state_raw: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    pas_amplitude: np.ndarray,
    pdp_amplitude: np.ndarray,
    valid: np.ndarray,
    canonical_query_ids: np.ndarray,
    canonical_fold: np.ndarray,
    truth_all: np.ndarray,
    surface: np.lib.npyio.NpzFile,
    device: torch.device,
    force: bool,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    result_path, metrics_path, checkpoint_path = fold_result_paths(outer, mode)
    if result_path.is_file() and metrics_path.is_file() and not force:
        record = json.loads(result_path.read_text(encoding="utf-8"))
        cached = np.load(metrics_path, allow_pickle=False)
        return record, {key: cached[key] for key in cached.files}

    set_seed((config.seed + outer * 1009 + (0 if mode == "actual" else 500_003)) % (2**32 - 1))
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    fit_ids = folds[f"fold{outer}_fit"].astype(np.int64)
    context_config = ContextConfig()
    train_table = build_train_contexts(
        outer, positions, site, valid, fit_ids, context_config
    )
    held_table = validation_context(outer, canonical_query_ids, canonical_fold)
    held_ids = held_table.query_ids.astype(np.int64)
    strict_rows = np.flatnonzero(canonical_fold == outer)
    if not np.array_equal(held_ids, canonical_query_ids[strict_rows]):
        raise RuntimeError(f"fold {outer} canonical query order mismatch")

    state_mean = np.mean(state_raw[fit_ids], axis=0, dtype=np.float64).astype(np.float32)
    state_std = np.maximum(
        np.std(state_raw[fit_ids], axis=0, dtype=np.float64).astype(np.float32), 1e-4
    )
    state = np.clip((state_raw - state_mean) / state_std, -8.0, 8.0).astype(np.float32)
    guard = 2
    basis_donor = train_table.donor_ids[:, guard].astype(np.int64)
    basis_weight = train_table.weights[:, guard].astype(np.float32)
    anchor_pas = e141.anchor_amplitude_numpy(
        pas_amplitude,
        basis_donor,
        basis_weight,
        (e141.MP, e141.MH, e141.MV, e141.N_UE),
        (1, 2, 3),
    )
    anchor_pdp = e141.anchor_amplitude_numpy(
        pdp_amplitude,
        basis_donor,
        basis_weight,
        (e141.MP, e141.N_UE, e141.S_SC),
        (3,),
    )
    train_ids = train_table.query_ids.astype(np.int64)
    pas_basis = e141.fit_residual_basis(
        pas_amplitude[train_ids],
        anchor_pas,
        config.pas_rank,
        config.seed + outer * 17,
    )
    pdp_basis = e141.fit_residual_basis(
        pdp_amplitude[train_ids],
        anchor_pdp,
        config.pdp_rank,
        config.seed + outer * 19,
    )
    donor_code = np.concatenate(
        (
            e141.encode_input(pas_amplitude, pas_basis),
            e141.encode_input(pdp_amplitude, pdp_basis),
        ),
        axis=1,
    ).astype(np.float32)

    global_source = (
        np.arange(len(positions), dtype=np.int64)
        if mode == "actual"
        else surface["control_source"].astype(np.int64)
    )
    raw_token = surface["token"].astype(np.float32)
    raw_mask = surface["mask"].astype(bool)
    fit_sources = global_source[train_ids]
    object_id, object_count, object_stats = compact_identifiers(
        surface["object_id"].astype(np.int32),
        raw_mask,
        fit_sources,
        config.minimum_object_support,
    )
    facet_id, facet_count, facet_stats = compact_identifiers(
        surface["facet_id"].astype(np.int32),
        raw_mask,
        fit_sources,
        config.minimum_facet_support,
    )
    token_rows = raw_token[fit_sources]
    token_mask = raw_mask[fit_sources]
    flat_fit = token_rows[token_mask]
    token_mean = flat_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    token_std = np.maximum(
        flat_fit.std(axis=0, dtype=np.float64).astype(np.float32), 1e-4
    )
    token = np.clip((raw_token - token_mean) / token_std, -8.0, 8.0).astype(np.float32)
    token[~raw_mask] = 0.0

    data = FoldData(
        state=torch.as_tensor(state, device=device),
        position=torch.as_tensor(positions, device=device),
        site=torch.as_tensor(site, dtype=torch.long, device=device),
        pas_amplitude=torch.as_tensor(pas_amplitude, device=device),
        pdp_amplitude=torch.as_tensor(pdp_amplitude, device=device),
        donor_code=torch.as_tensor(donor_code, device=device),
        pas_components=torch.as_tensor(pas_basis.components, device=device),
        pdp_components=torch.as_tensor(pdp_basis.components, device=device),
        pas_scale=torch.as_tensor(pas_basis.residual_scale, device=device),
        pdp_scale=torch.as_tensor(pdp_basis.residual_scale, device=device),
        token=torch.as_tensor(token, device=device),
        object_id=torch.as_tensor(object_id, dtype=torch.long, device=device),
        facet_id=torch.as_tensor(facet_id, dtype=torch.long, device=device),
        mask=torch.as_tensor(raw_mask, dtype=torch.bool, device=device),
        source=torch.as_tensor(global_source, dtype=torch.long, device=device),
    )
    code_dim = config.pas_rank + config.pdp_rank
    query_dim = state.shape[1] + 3 * code_dim + 4
    model = SurfaceInverseRenderer(
        query_dim, object_count, facet_count, config
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_context = e141.tensor_context(train_table, device)
    pas_mean = torch.as_tensor(pas_basis.input_mean, device=device)
    pas_input_scale = torch.as_tensor(pas_basis.input_scale, device=device)
    pdp_mean = torch.as_tensor(pdp_basis.input_mean, device=device)
    pdp_input_scale = torch.as_tensor(pdp_basis.input_scale, device=device)
    rng = np.random.default_rng(config.seed + outer * 2003 + (0 if mode == "actual" else 700_001))
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(train_ids))
        epoch_rows: list[tuple[float, float, float, float, float]] = []
        for start in range(0, len(order), config.batch_size):
            local = order[start : start + config.batch_size]
            rows = torch.as_tensor(local, dtype=torch.long, device=device)
            guards = torch.as_tensor(
                rng.integers(0, len(context_config.guard_radii_m), size=len(local)),
                dtype=torch.long,
                device=device,
            )
            predicted_pas, predicted_pdp, coefficient, query_ids, anchor = forward_batch(
                model,
                train_context,
                rows,
                guards,
                data,
                pas_mean,
                pas_input_scale,
                pdp_mean,
                pdp_input_scale,
            )
            truth_pas = data.pas_amplitude[query_ids]
            truth_pdp = data.pdp_amplitude[query_ids]
            pas_cosine = e141.amplitude_cosine(
                predicted_pas,
                truth_pas,
                (e141.MP, e141.MH, e141.MV, e141.N_UE),
                (1, 2, 3),
            )
            pdp_cosine = e141.amplitude_cosine(
                predicted_pdp,
                truth_pdp,
                (e141.MP, e141.N_UE, e141.S_SC),
                (3,),
            )
            pas_target = e141.residual_coefficients(
                truth_pas,
                anchor[:, : e141.PAS_SIZE],
                data.pas_components,
                data.pas_scale,
            )
            pdp_target = e141.residual_coefficients(
                truth_pdp,
                anchor[:, e141.PAS_SIZE :],
                data.pdp_components,
                data.pdp_scale,
            )
            target = torch.cat((pas_target, pdp_target), dim=1)
            coefficient_loss = (coefficient - target).square().mean()
            action = coefficient.square().mean()
            loss = 1.0 - 0.5 * (pas_cosine.mean() + pdp_cosine.mean())
            loss = loss + config.coefficient_weight * coefficient_loss
            loss = loss + config.action_regularization * action
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            epoch_rows.append(
                (
                    float(loss.detach().cpu()),
                    float(pas_cosine.mean().detach().cpu()),
                    float(pdp_cosine.mean().detach().cpu()),
                    float(coefficient_loss.detach().cpu()),
                    float(torch.sqrt(action.detach()).cpu()),
                )
            )
        row = {
            "epoch": epoch,
            "loss": float(np.mean([item[0] for item in epoch_rows])),
            "pas": float(np.mean([item[1] for item in epoch_rows])),
            "pdp": float(np.mean([item[2] for item in epoch_rows])),
            "coefficient_mse": float(np.mean([item[3] for item in epoch_rows])),
            "action_rms": float(np.mean([item[4] for item in epoch_rows])),
        }
        history.append(row)
        if epoch == 1 or epoch % 4 == 0 or epoch == config.epochs:
            print(
                f"E145 fold={outer} mode={mode} epoch={epoch}/{config.epochs} "
                f"train={row['pas']:.6f}/{row['pdp']:.6f} "
                f"coef={row['coefficient_mse']:.5f} action={row['action_rms']:.4f}",
                flush=True,
            )

    prediction = predict(
        model,
        held_table,
        data,
        pas_basis,
        pdp_basis,
        config.batch_size,
        device,
    )
    truth_pas = pas_amplitude[held_ids]
    truth_pdp = pdp_amplitude[held_ids]
    endpoint = e141.endpoint_summary(
        prediction["pas"], prediction["pdp"], truth_pas, truth_pdp
    )
    anchor_endpoint = e141.endpoint_summary(
        prediction["anchor_pas"], prediction["anchor_pdp"], truth_pas, truth_pdp
    )
    predicted_pas = prediction["pas"].reshape(
        -1, e141.MP, e141.MH, e141.MV, e141.N_UE
    ) ** 2
    predicted_pdp = prediction["pdp"].reshape(
        -1, e141.MP, e141.N_UE, e141.S_SC
    ) ** 2
    anchor_pas_probability = prediction["anchor_pas"].reshape(
        -1, e141.MP, e141.MH, e141.MV, e141.N_UE
    ) ** 2
    anchor_pdp_probability = prediction["anchor_pdp"].reshape(
        -1, e141.MP, e141.N_UE, e141.S_SC
    ) ** 2
    v7_point = e141.complete_point_metrics(
        e141.V7_CHANNEL,
        strict_rows,
        held_ids,
        truth_all,
        device,
        config.batch_size,
    )
    anchor_point = e141.complete_point_metrics(
        e141.V7_CHANNEL,
        strict_rows,
        held_ids,
        truth_all,
        device,
        config.batch_size,
        anchor_pas_probability,
        anchor_pdp_probability,
        config,
    )
    model_point = e141.complete_point_metrics(
        e141.V7_CHANNEL,
        strict_rows,
        held_ids,
        truth_all,
        device,
        config.batch_size,
        predicted_pas,
        predicted_pdp,
        config,
    )
    methods = {
        "v7": statistics(*v7_point),
        "anchor": statistics(*anchor_point),
        "model": statistics(*model_point),
    }
    record: dict[str, object] = {
        "experiment": VERSION,
        "fold": outer,
        "mode": mode,
        "fit_count": int(len(fit_ids)),
        "train_count": int(len(train_ids)),
        "held_count": int(len(held_ids)),
        "state_dim": int(state.shape[1]),
        "query_dim": int(query_dim),
        "object": object_stats,
        "facet": facet_stats,
        "basis": {
            "pas_rank": pas_basis.rank,
            "pdp_rank": pdp_basis.rank,
            "pas_explained_energy": pas_basis.explained_energy,
            "pdp_explained_energy": pdp_basis.explained_energy,
        },
        "endpoint": {"anchor": anchor_endpoint, "model": endpoint},
        "complete": methods,
        "delta_to_v7": float(methods["model"]["score"] - methods["v7"]["score"]),
        "training": history,
        "checkpoint": PATHS.relative(checkpoint_path),
        "seconds": float(time.time() - started),
    }
    torch.save(
        {
            "experiment": VERSION,
            "fold": outer,
            "mode": mode,
            "config": asdict(config),
            "state_dict": model.state_dict(),
            "state_mean": state_mean,
            "state_std": state_std,
            "token_mean": token_mean,
            "token_std": token_std,
            "pas_basis": asdict(pas_basis),
            "pdp_basis": asdict(pdp_basis),
            "official_h_only": True,
            "sionna_required_for_inference": False,
        },
        checkpoint_path,
    )
    arrays: dict[str, np.ndarray] = {
        "query_ids": held_ids.astype(np.int32),
        "canonical_rows": strict_rows.astype(np.int32),
        "predicted_pas": predicted_pas.astype(np.float16),
        "predicted_pdp": predicted_pdp.astype(np.float16),
        "anchor_pas": anchor_pas_probability.astype(np.float16),
        "anchor_pdp": anchor_pdp_probability.astype(np.float16),
        "coefficient": prediction["coefficient"].astype(np.float16),
    }
    for name, point in (("v7", v7_point), ("anchor", anchor_point), ("model", model_point)):
        for metric, value in zip(("pas", "pdp", "error", "energy"), point):
            arrays[f"{name}_{metric}"] = value
    np.savez_compressed(metrics_path, **arrays)
    result_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    del model, data
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record, arrays


def aggregate_records(
    records: list[dict[str, object]], arrays: list[dict[str, np.ndarray]]
) -> dict[str, object]:
    by_mode: dict[str, dict[str, object]] = {}
    for mode in sorted({str(record["mode"]) for record in records}):
        selected = [
            (record, value)
            for record, value in zip(records, arrays)
            if record["mode"] == mode
        ]
        methods: dict[str, object] = {}
        for method in ("v7", "anchor", "model"):
            point = tuple(
                np.concatenate([value[f"{method}_{metric}"] for _, value in selected])
                for metric in ("pas", "pdp", "error", "energy")
            )
            methods[method] = statistics(*point)
        by_mode[mode] = {
            "methods": methods,
            "fold_delta": [float(record["delta_to_v7"]) for record, _ in selected],
            "folds": [int(record["fold"]) for record, _ in selected],
        }
    return by_mode


def main() -> None:
    args = parse_args()
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    if not folds or any(value not in range(5) for value in folds):
        raise ValueError("--folds must contain values in [0,4]")
    if set(modes) - {"actual", "control"}:
        raise ValueError("--modes accepts only actual,control")
    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        pas_rank=args.pas_rank,
        pdp_rank=args.pdp_rank,
    )
    device = torch.device(args.device)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cache_report = build_surface_cache(args.force_surface_cache)
    surface = np.load(SURFACE_CACHE, allow_pickle=False)
    state, positions, site = e141.build_state_features()
    pas_amplitude, pdp_amplitude, valid = e141.probability_targets()
    metadata = np.load(OOF_META, allow_pickle=False)
    canonical_query_ids = metadata["query_ids"].astype(np.int64)
    canonical_fold = metadata["fold_id"].astype(np.int8)
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    records: list[dict[str, object]] = []
    arrays: list[dict[str, np.ndarray]] = []
    for outer in folds:
        for mode in modes:
            record, values = train_mode(
                outer,
                mode,
                config,
                state,
                positions,
                site,
                pas_amplitude,
                pdp_amplitude,
                valid,
                canonical_query_ids,
                canonical_fold,
                truth_all,
                surface,
                device,
                args.force_train,
            )
            records.append(record)
            arrays.append(values)
            print(
                f"E145 fold={outer} mode={mode} "
                f"model={record['complete']['model']['score']:.6f} "
                f"v7={record['complete']['v7']['score']:.6f} "
                f"delta={record['delta_to_v7']:+.6f}",
                flush=True,
            )

    aggregate = aggregate_records(records, arrays)
    fold_gate: dict[str, object] = {}
    for outer in folds:
        actual = next(
            (record for record in records if record["fold"] == outer and record["mode"] == "actual"),
            None,
        )
        control = next(
            (record for record in records if record["fold"] == outer and record["mode"] == "control"),
            None,
        )
        if actual is None:
            continue
        actual_model = actual["complete"]["model"]
        actual_v7 = actual["complete"]["v7"]
        control_score = (
            float(control["complete"]["model"]["score"])
            if control is not None
            else float("nan")
        )
        fold_gate[str(outer)] = {
            "actual_gain": float(actual["delta_to_v7"]),
            "actual_minus_control": float(actual_model["score"] - control_score),
            "pas_non_worse": bool(actual_model["pas"] >= actual_v7["pas"]),
            "pdp_non_worse": bool(actual_model["pdp"] >= actual_v7["pdp"]),
            "nmse_non_worse": bool(actual_model["nmse"] <= actual_v7["nmse"]),
            "passed": bool(
                control is not None
                and float(actual["delta_to_v7"]) >= config.required_screen_gain
                and float(actual_model["score"]) > control_score
                and actual_model["pas"] >= actual_v7["pas"]
                and actual_model["pdp"] >= actual_v7["pdp"]
                and actual_model["nmse"] <= actual_v7["nmse"]
            ),
        }
    complete_five = set(folds) == set(range(5)) and set(modes) == {"actual", "control"}
    fivefold_gate: dict[str, object] = {"evaluated": complete_five}
    if complete_five:
        actual = aggregate["actual"]
        control = aggregate["control"]
        model = actual["methods"]["model"]
        v7 = actual["methods"]["v7"]
        fivefold_gate.update(
            {
                "actual_score": float(model["score"]),
                "required_score": float(v7["score"] + config.required_full_gain),
                "positive_folds": int(
                    np.sum(np.asarray(actual["fold_delta"], dtype=np.float64) > 0.0)
                ),
                "actual_minus_control": float(
                    model["score"] - control["methods"]["model"]["score"]
                ),
                "above_v12": bool(model["score"] > V12_SCORE),
                "passed": bool(
                    model["score"] >= v7["score"] + config.required_full_gain
                    and np.all(np.asarray(actual["fold_delta"], dtype=np.float64) > 0.0)
                    and model["score"] > control["methods"]["model"]["score"]
                    and model["pas"] >= v7["pas"]
                    and model["pdp"] >= v7["pdp"]
                    and model["nmse"] <= v7["nmse"]
                ),
            }
        )
    report = {
        "experiment": VERSION,
        "status": "fivefold_complete" if complete_five else "screen_complete",
        "total_goal": (
            "End-to-end official-input AI with online score above 0.70; local complete "
            f"score must first exceed V12={V12_SCORE:.6f}."
        ),
        "primary_hypothesis": (
            "Official H can inversely identify a scene-shared effective radio field on "
            "map surfaces; ordered BS-to-UE surface sequences can then render a spectral "
            "correction that generalizes across spatial holes."
        ),
        "official_inputs_only": True,
        "sionna_or_v12_data_read": False,
        "v7_role": "Official-data-only complex carrier and frozen deployment baseline.",
        "config": asdict(config),
        "surface_cache": cache_report,
        "folds": records,
        "aggregate": aggregate,
        "screen_gate": fold_gate,
        "fivefold_gate": fivefold_gate,
        "decision": {
            "expand_after_fold3": bool(fold_gate.get("3", {}).get("passed", False)),
            "generate_candidate": bool(fivefold_gate.get("passed", False)),
        },
        "outputs": {
            "surface_cache": PATHS.relative(SURFACE_CACHE),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
