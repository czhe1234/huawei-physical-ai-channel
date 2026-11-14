"""Turn sampled route topology into compact per-point physical PathTokens."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import warp as wp

from project_paths import PATHS

from .config import (
    INTERACTION_REFLECTION,
    INTERACTION_SCATTER,
    INTERACTION_TRANSMISSION,
    LIGHT_SPEED_M_S,
    PhysicalPriorConfig,
    SOURCE_BASE,
    SOURCE_SCATTER,
)
from .geometry import GeometryData
from .kernels import MAX_TRACE_DEPTH, collect_segment_hits
from .tracer import ARTIFACT_DIR, route_bank_path, scatter_bank_path


RAW_DIR = ARTIFACT_DIR / "raw_tokens_v2"
FINAL_DIR = ARTIFACT_DIR / "path_tokens"


TOKEN_DTYPES: dict[str, Any] = {
    "mask": np.bool_,
    "source": np.int8,
    "tau": np.float32,
    "path_length": np.float32,
    "theta_t": np.float32,
    "phi_t": np.float32,
    "theta_r": np.float32,
    "phi_r": np.float32,
    "relative_energy": np.float32,
    "presence": np.uint8,
    "interaction_count": np.uint8,
    "interactions": np.uint8,
    "surface_family": np.int32,
    "vertices": np.float32,
    "geometry_confidence": np.float32,
    "incidence": np.float32,
    "seed": np.int32,
}


@dataclass
class PathCandidate:
    source: int
    seed: int
    presence: int
    path_length: float
    departure: np.ndarray
    arrival: np.ndarray
    energy: float
    confidence: float
    interactions: np.ndarray
    families: np.ndarray
    vertices: np.ndarray
    incidence: np.ndarray

    @property
    def interaction_count(self) -> int:
        return int(len(self.interactions))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def raw_shard_path(site: int, shard: int) -> Path:
    return RAW_DIR / f"site{site}" / f"shard{shard:02d}.npz"


def raw_shard_report_path(site: int, shard: int) -> Path:
    return raw_shard_path(site, shard).with_suffix(".json")


def direction_angles(direction: np.ndarray) -> tuple[float, float]:
    unit = np.asarray(direction, dtype=np.float64)
    unit /= max(float(np.linalg.norm(unit)), 1e-12)
    return (
        float(np.arccos(np.clip(unit[2], -1.0, 1.0))),
        float(np.arctan2(unit[1], unit[0])),
    )


def _family_inside(
    point: np.ndarray,
    family: np.ndarray,
    geometry: GeometryData,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    safe = np.maximum(family, 0)
    delta = point - geometry.family_origin[safe]
    u = np.einsum("ri,ri->r", delta, geometry.family_u[safe])
    v = np.einsum("ri,ri->r", delta, geometry.family_v[safe])
    low = geometry.family_uv_min[safe] - margin
    high = geometry.family_uv_max[safe] + margin
    inside = (u >= low[:, 0]) & (u <= high[:, 0]) & (v >= low[:, 1]) & (v <= high[:, 1])
    width = np.maximum(high - low, 1e-3)
    edge = np.minimum.reduce(
        (
            (u - low[:, 0]) / width[:, 0],
            (high[:, 0] - u) / width[:, 0],
            (v - low[:, 1]) / width[:, 1],
            (high[:, 1] - v) / width[:, 1],
        )
    )
    confidence = np.clip(4.0 * edge, 0.0, 1.0)
    return inside, confidence


def solve_specular_routes(
    bs: np.ndarray,
    ue: np.ndarray,
    route_family: np.ndarray,
    route_depth: np.ndarray,
    route_count: np.ndarray,
    route_energy: np.ndarray,
    route_score: np.ndarray,
    geometry: GeometryData,
    config: PhysicalPriorConfig,
) -> dict[str, np.ndarray]:
    """Solve ordered reflection planes by unfolding the receiver in reverse."""

    family = np.asarray(route_family, dtype=np.int32)
    depth = np.asarray(route_depth, dtype=np.int16)
    count = len(family)
    if count == 0:
        return {
            "valid": np.empty(0, dtype=bool),
            "points": np.empty((0, MAX_TRACE_DEPTH, 3), dtype=np.float32),
            "path_length": np.empty(0, dtype=np.float32),
            "departure": np.empty((0, 3), dtype=np.float32),
            "arrival": np.empty((0, 3), dtype=np.float32),
            "energy": np.empty(0, dtype=np.float32),
            "confidence": np.empty(0, dtype=np.float32),
            "incidence": np.empty((0, MAX_TRACE_DEPTH), dtype=np.float32),
        }
    safe = np.maximum(family, 0)
    active = np.arange(MAX_TRACE_DEPTH)[None] < depth[:, None]
    images = np.zeros((count, MAX_TRACE_DEPTH, 3), dtype=np.float64)
    image = np.broadcast_to(np.asarray(ue, dtype=np.float64), (count, 3)).copy()
    for step in range(MAX_TRACE_DEPTH - 1, -1, -1):
        rows = active[:, step]
        if not np.any(rows):
            continue
        local_family = safe[rows, step]
        normal = geometry.family_normal[local_family].astype(np.float64)
        offset = geometry.family_offset[local_family].astype(np.float64)
        distance = np.einsum("ri,ri->r", image[rows], normal) + offset
        image[rows] -= 2.0 * distance[:, None] * normal
        images[rows, step] = image[rows]

    points = np.zeros((count, MAX_TRACE_DEPTH, 3), dtype=np.float64)
    incidence = np.zeros((count, MAX_TRACE_DEPTH), dtype=np.float64)
    current = np.broadcast_to(np.asarray(bs, dtype=np.float64), (count, 3)).copy()
    valid = np.ones(count, dtype=bool)
    confidence = np.ones(count, dtype=np.float64)
    path_length = np.zeros(count, dtype=np.float64)
    reflection_gain = np.ones(count, dtype=np.float64)
    for step in range(MAX_TRACE_DEPTH):
        rows = active[:, step]
        if not np.any(rows):
            continue
        local_family = safe[rows, step]
        normal = geometry.family_normal[local_family].astype(np.float64)
        offset = geometry.family_offset[local_family].astype(np.float64)
        vector = images[rows, step] - current[rows]
        denominator = np.einsum("ri,ri->r", vector, normal)
        numerator = -(
            np.einsum("ri,ri->r", current[rows], normal) + offset
        )
        parameter = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=np.abs(denominator) > 1e-9,
        )
        point = current[rows] + parameter[:, None] * vector
        inside, edge_confidence = _family_inside(
            point, local_family, geometry, config.family_bounds_margin_m
        )
        local_valid = (
            np.isfinite(parameter)
            & (parameter > 1e-6)
            & (parameter < 1.0 - 1e-6)
            & inside
        )
        row_ids = np.flatnonzero(rows)
        valid[row_ids] &= local_valid
        confidence[row_ids] = np.minimum(confidence[row_ids], edge_confidence)
        segment = point - current[rows]
        segment_norm = np.linalg.norm(segment, axis=1)
        path_length[row_ids] += segment_norm
        incoming = segment / np.maximum(segment_norm[:, None], 1e-12)
        cosine = np.abs(np.einsum("ri,ri->r", incoming, normal))
        incidence[row_ids, step] = cosine
        reflection_gain[row_ids] *= 0.58 + 0.32 * (1.0 - cosine)
        points[row_ids, step] = point
        current[row_ids] = point

    final_segment = np.asarray(ue, dtype=np.float64)[None] - current
    final_norm = np.linalg.norm(final_segment, axis=1)
    path_length += final_norm
    direct = float(np.linalg.norm(np.asarray(ue) - np.asarray(bs)))
    valid &= np.isfinite(path_length)
    valid &= path_length >= direct - 1e-3
    valid &= path_length <= config.maximum_path_length_m
    first = np.where(
        (depth > 0)[:, None], points[:, 0] - np.asarray(bs)[None], np.asarray(ue)[None] - np.asarray(bs)[None]
    )
    arrival = final_segment
    prior = np.maximum(route_score.astype(np.float64), 1e-15)
    prior *= 1.0 + 0.04 * np.log1p(route_count.astype(np.float64))
    prior *= np.sqrt(np.maximum(route_energy.astype(np.float64), 1e-15))
    energy = (
        prior
        * reflection_gain
        / np.maximum(path_length * path_length, 1.0)
        * np.exp(-0.0015 * np.maximum(path_length - direct, 0.0))
    )
    return {
        "valid": valid,
        "points": points.astype(np.float32),
        "path_length": path_length.astype(np.float32),
        "departure": first.astype(np.float32),
        "arrival": arrival.astype(np.float32),
        "energy": energy.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "incidence": incidence.astype(np.float32),
    }


class SegmentCollector:
    """Collect interior mesh crossings while keeping the mesh resident on one GPU."""

    def __init__(self, geometry: GeometryData, device: str, config: PhysicalPriorConfig):
        wp.init()
        self.device = device
        self.config = config
        with wp.ScopedDevice(device):
            self.mesh = wp.Mesh(
                points=wp.array(geometry.vertices, dtype=wp.vec3, device=device),
                indices=wp.array(
                    geometry.faces.reshape(-1), dtype=wp.int32, device=device
                ),
            )
            self.face_family = wp.array(
                geometry.face_family, dtype=wp.int32, device=device
            )

    def collect(self, starts: np.ndarray, ends: np.ndarray) -> dict[str, np.ndarray]:
        count = len(starts)
        if count == 0:
            return {
                "count": np.empty(0, dtype=np.uint8),
                "family": np.empty((0, MAX_TRACE_DEPTH), dtype=np.int32),
                "vertices": np.empty((0, MAX_TRACE_DEPTH, 3), dtype=np.float32),
                "incidence": np.empty((0, MAX_TRACE_DEPTH), dtype=np.float32),
            }
        with wp.ScopedDevice(self.device):
            start_wp = wp.array(starts, dtype=wp.vec3, device=self.device)
            end_wp = wp.array(ends, dtype=wp.vec3, device=self.device)
            counts = wp.zeros(count, dtype=wp.uint8, device=self.device)
            family = wp.full(
                count * MAX_TRACE_DEPTH,
                value=-1,
                dtype=wp.int32,
                device=self.device,
            )
            vertices = wp.zeros(
                count * MAX_TRACE_DEPTH, dtype=wp.vec3, device=self.device
            )
            incidence = wp.zeros(
                count * MAX_TRACE_DEPTH, dtype=wp.float32, device=self.device
            )
            wp.launch(
                collect_segment_hits,
                dim=count,
                inputs=(
                    self.mesh.id,
                    self.face_family,
                    start_wp,
                    end_wp,
                    float(self.config.ray_epsilon_m),
                    int(self.config.maximum_interactions),
                ),
                outputs=(counts, family, vertices, incidence),
                device=self.device,
            )
            wp.synchronize_device(self.device)
            return {
                "count": counts.numpy(),
                "family": family.numpy().reshape(count, MAX_TRACE_DEPTH),
                "vertices": vertices.numpy().reshape(count, MAX_TRACE_DEPTH, 3),
                "incidence": incidence.numpy().reshape(count, MAX_TRACE_DEPTH),
            }


def _base_identity(candidate: PathCandidate) -> tuple[Any, ...]:
    theta, phi = direction_angles(candidate.departure)
    delay_m = candidate.path_length
    reflection_family = tuple(
        int(family)
        for interaction, family in zip(candidate.interactions, candidate.families)
        if interaction == INTERACTION_REFLECTION
    )
    return (
        int(round(theta * 80.0)),
        int(round(phi * 80.0)),
        int(round(delay_m * 2.0)),
        reflection_family,
    )


def _scatter_identity(candidate: PathCandidate) -> tuple[int, ...]:
    theta_t, phi_t = direction_angles(candidate.departure)
    theta_r, phi_r = direction_angles(candidate.arrival)
    return (
        int(round(theta_t * 12.0)),
        int(round(phi_t * 12.0)),
        int(round(theta_r * 12.0)),
        int(round(phi_r * 12.0)),
        int(round(candidate.path_length / 2.0)),
        min(candidate.interaction_count, 15),
    )


def _merge_base(candidates: list[PathCandidate], maximum: int) -> list[PathCandidate]:
    bank: dict[tuple[Any, ...], PathCandidate] = {}
    seeds: dict[tuple[Any, ...], set[int]] = {}
    for candidate in candidates:
        key = _base_identity(candidate)
        seeds.setdefault(key, set()).add(candidate.seed)
        previous = bank.get(key)
        if previous is None or candidate.energy > previous.energy:
            bank[key] = candidate
    output = []
    for key, candidate in bank.items():
        candidate.presence = len(seeds[key])
        output.append(candidate)
    output.sort(
        key=lambda item: item.energy * (1.0 + 0.05 * (item.presence - 1)),
        reverse=True,
    )
    return output[:maximum]


def _merge_scatter(candidates: list[PathCandidate], maximum: int) -> list[PathCandidate]:
    bank: dict[tuple[int, ...], PathCandidate] = {}
    for candidate in candidates:
        key = _scatter_identity(candidate)
        previous = bank.get(key)
        if previous is None or candidate.energy > previous.energy:
            bank[key] = candidate
    output = sorted(bank.values(), key=lambda item: item.energy, reverse=True)
    return output[:maximum]


def _polyline_length(
    start: np.ndarray, vertices: np.ndarray, end: np.ndarray
) -> float:
    points = np.concatenate(
        (
            np.asarray(start, dtype=np.float64).reshape(1, 3),
            np.asarray(vertices, dtype=np.float64).reshape(-1, 3),
            np.asarray(end, dtype=np.float64).reshape(1, 3),
        ),
        axis=0,
    )
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _path_length_is_consistent(
    candidate: PathCandidate,
    start: np.ndarray,
    end: np.ndarray,
    tolerance_m: float = 0.05,
) -> bool:
    reconstructed = _polyline_length(start, candidate.vertices, end)
    return abs(reconstructed - candidate.path_length) <= tolerance_m


class BaseRouteIndex:
    """Keep all route banks resident and add exhaustive one-surface image paths."""

    def __init__(
        self,
        site: int,
        geometry: GeometryData,
        config: PhysicalPriorConfig,
    ) -> None:
        self.site = int(site)
        self.geometry = geometry
        self.config = config
        self.banks: dict[int, dict[str, np.ndarray]] = {}
        for seed in config.seeds:
            path = route_bank_path(site, seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as cache:
                depth = cache["depth"].astype(np.uint8)
                keep = depth >= 2
                self.banks[int(seed)] = {
                    "family": cache["family"][keep].astype(np.int32),
                    "depth": depth[keep],
                    "count": cache["count"][keep].astype(np.int32),
                    "energy": cache["energy"][keep].astype(np.float32),
                    "score": cache["score"][keep].astype(np.float32),
                }
        family_ids = np.arange(geometry.family_count, dtype=np.int32)
        self.single_family = np.full(
            (geometry.family_count, MAX_TRACE_DEPTH), -1, dtype=np.int32
        )
        self.single_family[:, 0] = family_ids
        self.single_depth = np.ones(geometry.family_count, dtype=np.uint8)
        self.single_count = np.full(
            geometry.family_count, len(config.seeds), dtype=np.int32
        )
        area = np.maximum(geometry.family_area.astype(np.float64), 1e-6)
        median = max(float(np.median(area)), 1e-6)
        self.single_energy = np.ones(geometry.family_count, dtype=np.float32)
        self.single_score = np.clip(np.sqrt(area / median), 0.2, 5.0).astype(
            np.float32
        )

    @staticmethod
    def _records(
        solved: dict[str, np.ndarray],
        family: np.ndarray,
        depth: np.ndarray,
        seed: int,
        presence: int,
        maximum: int,
    ) -> list[PathCandidate]:
        valid = np.flatnonzero(solved["valid"])
        if len(valid) > maximum:
            local_energy = solved["energy"][valid]
            keep = np.argpartition(local_energy, len(local_energy) - maximum)[
                -maximum:
            ]
            valid = valid[keep]
        valid = valid[np.argsort(-solved["energy"][valid])]
        output: list[PathCandidate] = []
        for row in valid:
            local_depth = int(depth[row])
            output.append(
                PathCandidate(
                    source=SOURCE_BASE,
                    seed=seed,
                    presence=presence,
                    path_length=float(solved["path_length"][row]),
                    departure=solved["departure"][row],
                    arrival=solved["arrival"][row],
                    energy=float(solved["energy"][row]),
                    confidence=float(solved["confidence"][row]),
                    interactions=np.full(
                        local_depth, INTERACTION_REFLECTION, dtype=np.uint8
                    ),
                    families=family[row, :local_depth].astype(np.int32),
                    vertices=solved["points"][row, :local_depth],
                    incidence=solved["incidence"][row, :local_depth],
                )
            )
        return output

    def candidates(self, bs: np.ndarray, ue: np.ndarray) -> list[PathCandidate]:
        config = self.config
        output: list[PathCandidate] = []
        direct_vector = np.asarray(ue - bs, dtype=np.float32)
        direct_length = float(np.linalg.norm(direct_vector))
        output.append(
            PathCandidate(
                source=SOURCE_BASE,
                seed=-1,
                presence=len(config.seeds),
                path_length=direct_length,
                departure=direct_vector,
                arrival=direct_vector,
                energy=1.0 / max(direct_length * direct_length, 1.0),
                confidence=1.0,
                interactions=np.empty(0, dtype=np.uint8),
                families=np.empty(0, dtype=np.int32),
                vertices=np.empty((0, 3), dtype=np.float32),
                incidence=np.empty(0, dtype=np.float32),
            )
        )
        single = solve_specular_routes(
            bs,
            ue,
            self.single_family,
            self.single_depth,
            self.single_count,
            self.single_energy,
            self.single_score,
            self.geometry,
            config,
        )
        output.extend(
            self._records(
                single,
                self.single_family,
                self.single_depth,
                seed=-1,
                presence=len(config.seeds),
                maximum=config.base_pool * 2,
            )
        )
        for seed, bank in self.banks.items():
            solved = solve_specular_routes(
                bs,
                ue,
                bank["family"],
                bank["depth"],
                bank["count"],
                bank["energy"],
                bank["score"],
                self.geometry,
                config,
            )
            output.extend(
                self._records(
                    solved,
                    bank["family"],
                    bank["depth"],
                    seed=seed,
                    presence=1,
                    maximum=config.base_per_seed * 4,
                )
            )
        return _merge_base(output, config.base_pool)


class ScatterIndex:
    def __init__(self, site: int, seed: int):
        path = scatter_bank_path(site, seed)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.cache = np.load(path, allow_pickle=False)
        depth = self.cache["depth"].astype(np.int64)
        rows = np.arange(len(depth), dtype=np.int64)
        self.point = self.cache["vertices"][rows, depth - 1].astype(np.float32)
        self.tree = cKDTree(self.point) if len(self.point) else None
        self.seed = int(seed)

    def candidates(
        self, ue: np.ndarray, config: PhysicalPriorConfig
    ) -> list[PathCandidate]:
        if self.tree is None:
            return []
        query_count = min(len(self.point), config.scatter_per_seed * 8)
        distance, rows = self.tree.query(ue, k=query_count)
        distance = np.atleast_1d(distance).astype(np.float64)
        rows = np.atleast_1d(rows).astype(np.int64)
        depth = self.cache["depth"][rows].astype(np.int64)
        total_length = self.cache["path_length"][rows].astype(np.float64) + distance
        energy = self.cache["energy"][rows].astype(np.float64)
        score = energy / (1.0 + 0.001 * np.square(distance))
        score *= np.exp(-0.0015 * distance)
        keep = min(config.scatter_per_seed, len(rows))
        selected = np.argpartition(score, len(score) - keep)[-keep:] if keep < len(rows) else np.arange(len(rows))
        selected = selected[np.argsort(-score[selected])]
        output: list[PathCandidate] = []
        for local in selected:
            row = int(rows[local])
            local_depth = int(depth[local])
            last = self.point[row]
            output.append(
                PathCandidate(
                    source=SOURCE_SCATTER,
                    seed=self.seed,
                    presence=1,
                    path_length=float(total_length[local]),
                    departure=self.cache["initial_direction"][row].astype(np.float32),
                    arrival=np.asarray(ue - last, dtype=np.float32),
                    energy=float(score[local]),
                    confidence=float(
                        np.clip(
                            math.sqrt(max(float(self.cache["energy"][row]), 0.0))
                            * math.exp(-0.002 * float(distance[local])),
                            0.0,
                            1.0,
                        )
                    ),
                    interactions=self.cache["interaction"][row, :local_depth].astype(
                        np.uint8
                    ),
                    families=self.cache["family"][row, :local_depth].astype(np.int32),
                    vertices=self.cache["vertices"][row, :local_depth].astype(
                        np.float32
                    ),
                    incidence=self.cache["incidence"][row, :local_depth].astype(
                        np.float32
                    ),
                )
            )
        return output


def _enrich_base(
    candidates: list[PathCandidate],
    bs: np.ndarray,
    ue: np.ndarray,
    collector: SegmentCollector,
    config: PhysicalPriorConfig,
) -> list[PathCandidate]:
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    owner: list[tuple[int, int]] = []
    for candidate_id, candidate in enumerate(candidates):
        route = [np.asarray(bs, dtype=np.float32), *candidate.vertices, np.asarray(ue, dtype=np.float32)]
        for segment in range(len(route) - 1):
            starts.append(route[segment])
            ends.append(route[segment + 1])
            owner.append((candidate_id, segment))
    hits = collector.collect(np.asarray(starts, dtype=np.float32), np.asarray(ends, dtype=np.float32))
    grouped: dict[tuple[int, int], int] = {
        key: row for row, key in enumerate(owner)
    }
    output: list[PathCandidate] = []
    for candidate_id, candidate in enumerate(candidates):
        interaction: list[int] = []
        family: list[int] = []
        vertices: list[np.ndarray] = []
        incidence: list[float] = []
        reflection_index = 0
        segments = candidate.interaction_count + 1
        transmission_count = 0
        for segment in range(segments):
            row = grouped[(candidate_id, segment)]
            local_count = int(hits["count"][row])
            for local in range(local_count):
                if len(interaction) >= config.maximum_interactions:
                    break
                interaction.append(INTERACTION_TRANSMISSION)
                family.append(int(hits["family"][row, local]))
                vertices.append(hits["vertices"][row, local])
                incidence.append(float(hits["incidence"][row, local]))
                transmission_count += 1
            if segment < candidate.interaction_count and len(interaction) < config.maximum_interactions:
                interaction.append(INTERACTION_REFLECTION)
                family.append(int(candidate.families[reflection_index]))
                vertices.append(candidate.vertices[reflection_index])
                incidence.append(float(candidate.incidence[reflection_index]))
                reflection_index += 1
        candidate.interactions = np.asarray(interaction, dtype=np.uint8)
        candidate.families = np.asarray(family, dtype=np.int32)
        candidate.vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        candidate.incidence = np.asarray(incidence, dtype=np.float32)
        candidate.energy *= 0.58 ** transmission_count
        candidate.confidence *= 0.94 ** transmission_count
        if len(interaction) >= config.maximum_interactions:
            candidate.confidence *= 0.75
        if not _path_length_is_consistent(candidate, bs, ue):
            continue
        output.append(candidate)
    return output


def _enrich_scatter(
    candidates: list[PathCandidate],
    bs: np.ndarray,
    ue: np.ndarray,
    collector: SegmentCollector,
    config: PhysicalPriorConfig,
) -> list[PathCandidate]:
    if not candidates:
        return []
    starts = np.asarray([candidate.vertices[-1] for candidate in candidates], dtype=np.float32)
    ends = np.broadcast_to(np.asarray(ue, dtype=np.float32), starts.shape).copy()
    hits = collector.collect(starts, ends)
    for row, candidate in enumerate(candidates):
        available = config.maximum_interactions - candidate.interaction_count
        local_count = min(int(hits["count"][row]), max(available, 0))
        if local_count:
            candidate.interactions = np.concatenate(
                (
                    candidate.interactions,
                    np.full(local_count, INTERACTION_TRANSMISSION, dtype=np.uint8),
                )
            )
            candidate.families = np.concatenate(
                (candidate.families, hits["family"][row, :local_count].astype(np.int32))
            )
            candidate.vertices = np.concatenate(
                (candidate.vertices, hits["vertices"][row, :local_count].astype(np.float32)), axis=0
            )
            candidate.incidence = np.concatenate(
                (candidate.incidence, hits["incidence"][row, :local_count].astype(np.float32))
            )
            candidate.energy *= 0.58 ** local_count
            candidate.confidence *= 0.94 ** local_count
        if int(hits["count"][row]) > local_count:
            candidate.confidence *= 0.75
    return [
        candidate
        for candidate in candidates
        if _path_length_is_consistent(candidate, bs, ue)
    ]


def empty_token_arrays(count: int, candidates: int, depth: int) -> dict[str, np.ndarray]:
    shape = (count, candidates)
    return {
        "mask": np.zeros(shape, dtype=np.bool_),
        "source": np.full(shape, -1, dtype=np.int8),
        "tau": np.zeros(shape, dtype=np.float32),
        "path_length": np.zeros(shape, dtype=np.float32),
        "theta_t": np.zeros(shape, dtype=np.float32),
        "phi_t": np.zeros(shape, dtype=np.float32),
        "theta_r": np.zeros(shape, dtype=np.float32),
        "phi_r": np.zeros(shape, dtype=np.float32),
        "relative_energy": np.zeros(shape, dtype=np.float32),
        "presence": np.zeros(shape, dtype=np.uint8),
        "interaction_count": np.zeros(shape, dtype=np.uint8),
        "interactions": np.zeros(shape + (depth,), dtype=np.uint8),
        "surface_family": np.full(shape + (depth,), -1, dtype=np.int32),
        "vertices": np.zeros(shape + (depth, 3), dtype=np.float32),
        "geometry_confidence": np.zeros(shape, dtype=np.float32),
        "incidence": np.zeros(shape + (depth,), dtype=np.float32),
        "seed": np.full(shape, -1, dtype=np.int32),
    }


def write_candidate(
    arrays: dict[str, np.ndarray], row: int, slot: int, candidate: PathCandidate
) -> None:
    arrays["mask"][row, slot] = True
    arrays["source"][row, slot] = candidate.source
    arrays["tau"][row, slot] = candidate.path_length / LIGHT_SPEED_M_S
    arrays["path_length"][row, slot] = candidate.path_length
    theta_t, phi_t = direction_angles(candidate.departure)
    theta_r, phi_r = direction_angles(candidate.arrival)
    arrays["theta_t"][row, slot] = theta_t
    arrays["phi_t"][row, slot] = phi_t
    arrays["theta_r"][row, slot] = theta_r
    arrays["phi_r"][row, slot] = phi_r
    arrays["presence"][row, slot] = min(candidate.presence, 255)
    depth = min(candidate.interaction_count, arrays["interactions"].shape[2])
    arrays["interaction_count"][row, slot] = depth
    arrays["interactions"][row, slot, :depth] = candidate.interactions[:depth]
    arrays["surface_family"][row, slot, :depth] = candidate.families[:depth]
    arrays["vertices"][row, slot, :depth] = candidate.vertices[:depth]
    arrays["incidence"][row, slot, :depth] = candidate.incidence[:depth]
    arrays["geometry_confidence"][row, slot] = candidate.confidence
    arrays["seed"][row, slot] = candidate.seed
    arrays["relative_energy"][row, slot] = max(candidate.energy, 0.0)


def build_raw_token_shard(
    *,
    positions: np.ndarray,
    point_ids: np.ndarray,
    site: int,
    bs_positions: np.ndarray,
    geometry: GeometryData,
    config: PhysicalPriorConfig,
    shard: int,
    shard_size: int,
    device: str,
    force: bool = False,
) -> dict[str, Any]:
    output_path = raw_shard_path(site, shard)
    report_path = raw_shard_report_path(site, shard)
    if output_path.is_file() and report_path.is_file() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "complete":
            return {"reused": True, **report}
    start = shard * shard_size
    end = min(start + shard_size, len(point_ids))
    selected_ids = point_ids[start:end]
    if len(selected_ids) == 0:
        raise ValueError(f"empty token shard site={site} shard={shard}")
    arrays = empty_token_arrays(
        len(selected_ids), config.raw_candidate_count, config.maximum_interactions
    )
    collector = SegmentCollector(geometry, device, config)
    base_index = BaseRouteIndex(site, geometry, config)
    scatter_indices = {
        seed: ScatterIndex(site, seed) for seed in config.seeds
    }
    started = time.time()
    base_count: list[int] = []
    scatter_count: list[int] = []
    for local_row, point_id in enumerate(selected_ids):
        ue = positions[int(point_id)]
        bs = bs_positions[site]
        base = base_index.candidates(bs, ue)
        base = _enrich_base(base, bs, ue, collector, config)
        scatter_candidates: list[PathCandidate] = []
        for index in scatter_indices.values():
            scatter_candidates.extend(index.candidates(ue, config))
        scatter = _merge_scatter(scatter_candidates, config.scatter_pool)
        scatter = _enrich_scatter(scatter, bs, ue, collector, config)
        base = sorted(base, key=lambda item: item.energy, reverse=True)[
            : config.base_pool
        ]
        scatter = sorted(scatter, key=lambda item: item.energy, reverse=True)[
            : config.scatter_pool
        ]
        selected = [*base, *scatter]
        selected = selected[: config.raw_candidate_count]
        total_energy = sum(max(candidate.energy, 0.0) for candidate in selected)
        for slot, candidate in enumerate(selected):
            write_candidate(arrays, local_row, slot, candidate)
        if total_energy > 0.0:
            arrays["relative_energy"][local_row] /= np.float32(total_energy)
        base_count.append(len(base))
        scatter_count.append(len(scatter))
        if (local_row + 1) % 25 == 0 or local_row + 1 == len(selected_ids):
            print(
                f"physical tokens site={site} shard={shard} "
                f"{local_row + 1}/{len(selected_ids)}",
                flush=True,
            )
    atomic_npz(output_path, point_ids=selected_ids.astype(np.int64), **arrays)
    report = {
        "status": "complete",
        "site": int(site),
        "shard": int(shard),
        "device": device,
        "point_count": int(len(selected_ids)),
        "base_candidate_mean": float(np.mean(base_count)),
        "scatter_candidate_mean": float(np.mean(scatter_count)),
        "valid_candidate_mean": float(np.mean(arrays["mask"].sum(axis=1))),
        "seconds": float(time.time() - started),
        "output": PATHS.relative(output_path),
    }
    atomic_json(report_path, report)
    for index in scatter_indices.values():
        index.cache.close()
    return report


def assemble_raw_tokens(
    site_point_ids: dict[int, np.ndarray],
    positions: np.ndarray,
    site_id: np.ndarray,
    bs_positions: np.ndarray,
    config: PhysicalPriorConfig,
    shard_size: int,
    output_path: Path,
) -> dict[str, Any]:
    count = sum(len(ids) for ids in site_point_ids.values())
    arrays = empty_token_arrays(
        count, config.raw_candidate_count, config.maximum_interactions
    )
    coverage = np.zeros(count, dtype=np.uint8)
    for site, point_ids in site_point_ids.items():
        for shard in range(int(math.ceil(len(point_ids) / shard_size))):
            path = raw_shard_path(site, shard)
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as cache:
                ids = cache["point_ids"].astype(np.int64)
                for name in arrays:
                    arrays[name][ids] = cache[name]
                coverage[ids] += 1
    if not np.all(coverage == 1):
        raise RuntimeError(
            f"raw token coverage failure {np.unique(coverage, return_counts=True)}"
        )

    mask = arrays["mask"]
    interaction_count = arrays["interaction_count"].astype(np.int64)
    previous = np.broadcast_to(
        np.asarray(bs_positions, dtype=np.float32)[site_id][:, None, :],
        (count, config.raw_candidate_count, 3),
    ).copy()
    endpoint = np.broadcast_to(
        np.asarray(positions, dtype=np.float32)[:, None, :], previous.shape
    )
    reconstructed = np.zeros(mask.shape, dtype=np.float32)
    for step in range(config.maximum_interactions):
        active = mask & (interaction_count > step)
        current = arrays["vertices"][:, :, step]
        reconstructed[active] += np.linalg.norm(
            current[active] - previous[active], axis=1
        )
        previous[active] = current[active]
    reconstructed[mask] += np.linalg.norm(
        endpoint[mask] - previous[mask], axis=1
    )
    length_error = np.abs(reconstructed - arrays["path_length"])
    inconsistent = mask & (length_error > np.float32(0.05))
    inconsistent_count = int(np.sum(inconsistent))
    active_length_error = length_error[mask].copy()
    if inconsistent_count:
        reset_values: dict[str, int | float | bool] = {
            "mask": False,
            "source": -1,
            "tau": 0.0,
            "path_length": 0.0,
            "theta_t": 0.0,
            "phi_t": 0.0,
            "theta_r": 0.0,
            "phi_r": 0.0,
            "relative_energy": 0.0,
            "presence": 0,
            "interaction_count": 0,
            "interactions": 0,
            "surface_family": -1,
            "vertices": 0.0,
            "geometry_confidence": 0.0,
            "incidence": 0.0,
            "seed": -1,
        }
        for name, value in reset_values.items():
            arrays[name][inconsistent] = value
        energy_sum = arrays["relative_energy"].sum(axis=1, keepdims=True)
        np.divide(
            arrays["relative_energy"],
            energy_sum,
            out=arrays["relative_energy"],
            where=energy_sum > 0.0,
        )
    atomic_npz(output_path, **arrays)
    return {
        "status": "complete",
        "point_count": int(count),
        "candidate_count": int(config.raw_candidate_count),
        "valid_candidate_mean": float(np.mean(arrays["mask"].sum(axis=1))),
        "removed_inconsistent_candidates": inconsistent_count,
        "path_length_error_p99_m": float(np.quantile(active_length_error, 0.99)),
        "path_length_error_max_m": float(np.max(active_length_error)),
        "remaining_path_length_error_max_m": float(
            np.max(length_error[arrays["mask"]])
        ),
        "all_finite": bool(
            all(
                np.all(np.isfinite(value))
                for value in arrays.values()
                if np.issubdtype(value.dtype, np.floating)
            )
        ),
        "output": PATHS.relative(output_path),
    }
