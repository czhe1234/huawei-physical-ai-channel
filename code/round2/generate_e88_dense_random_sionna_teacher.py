#!/usr/bin/env python3
"""Generate compact dense-position Sionna path teachers for Round2-E88.

The generator uses the same scene and radio configuration as the validated E39
source, but stores only path-state tokens and scorer-sufficient PAS/PDP targets.
It is resumable at point-batch granularity and never writes full CFR tensors.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from generate_multiseed_sionna_dataset import (
    CONFIGS,
    SEEDS,
    path_batch,
    prepare_runtime,
    select_drjit_cuda_device,
    tx_target,
)
from generate_v5_sionna_spectral_library import (
    SCENE_XML,
    configure_materials,
    prepare_scene,
)
from project_paths import PATHS
from round2.data import load_positions


VERSION = "Round2-E88-dense-random-Sionna-path-teacher"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "dense_random_sionna_teacher"
REPORT_DIR = PATHS.analysis / "results" / "round2_dense_random_sionna_teacher"
REPORT = REPORT_DIR / "dense_random_sionna_teacher.json"

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-count", type=int, default=60_000)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--path-slots", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sampling-seed", type=int, default=2026080801)
    parser.add_argument("--selection-tag", default="dense_n60000_s3")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--force-sampling", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def digest_arrays(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def open_memmap(path: Path, dtype: Any, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        value = np.load(path, mmap_mode="r+")
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(
                f"cache mismatch for {path}: {value.shape}/{value.dtype} "
                f"!= {shape}/{np.dtype(dtype)}"
            )
        return value
    value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    value[...] = 0
    value.flush()
    return value


def _site_boundary(train_positions: np.ndarray) -> float:
    ordered = np.sort(train_positions[:, 1])
    gap = int(np.argmax(np.diff(ordered)))
    return float(0.5 * (ordered[gap] + ordered[gap + 1]))


def _sample_site_positions(
    *,
    site_value: int,
    count: int,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_site: np.ndarray,
    test_site: np.ndarray,
    boundary: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    train = train_positions[train_site == site_value].astype(np.float64)
    test = test_positions[test_site == site_value].astype(np.float64)
    anchors = np.concatenate((train, test), axis=0)
    anchor_tree = cKDTree(anchors[:, :2])
    graph_tree = cKDTree(anchors[:, :2])
    _, graph_neighbors = graph_tree.query(anchors[:, :2], k=min(16, len(anchors)))

    lower = anchors[:, :2].min(axis=0) - 10.0
    upper = anchors[:, :2].max(axis=0) + 10.0
    seen: set[tuple[int, int]] = set()
    output: list[np.ndarray] = []
    kinds: list[int] = []

    # Query-centered rings intentionally use only known coordinates, never query H.
    # They make the offline teacher dense around the same spatial support seen at test.
    while len(output) < count:
        proposal_count = min(max(4096, count - len(output)), 32768)
        choice = rng.random(proposal_count)
        candidate = np.empty((proposal_count, 2), dtype=np.float64)
        kind = np.empty(proposal_count, dtype=np.uint8)

        query_ring = choice < 0.40
        train_ring = (choice >= 0.40) & (choice < 0.80)
        graph = choice >= 0.80

        for mask, pool, code in (
            (query_ring, test, 0),
            (train_ring, train, 1),
        ):
            local_count = int(mask.sum())
            if local_count == 0:
                continue
            selected = rng.integers(0, len(pool), size=local_count)
            # Log-uniform radii cover both sub-meter continuity and 5-8m holes.
            radius = np.exp(
                rng.uniform(np.log(0.35), np.log(8.0), size=local_count)
            )
            angle = rng.uniform(0.0, 2.0 * np.pi, size=local_count)
            candidate[mask] = pool[selected, :2] + np.column_stack(
                (radius * np.cos(angle), radius * np.sin(angle))
            )
            kind[mask] = code

        graph_count = int(graph.sum())
        if graph_count:
            first = rng.integers(0, len(anchors), size=graph_count)
            neighbor_rank = rng.integers(1, graph_neighbors.shape[1], size=graph_count)
            second = graph_neighbors[first, neighbor_rank]
            fraction = rng.uniform(0.05, 0.95, size=graph_count)
            left = anchors[first, :2]
            right = anchors[second, :2]
            route = right - left
            length = np.maximum(np.linalg.norm(route, axis=1), 1e-6)
            perpendicular = np.column_stack((-route[:, 1], route[:, 0])) / length[:, None]
            jitter = np.clip(rng.normal(0.0, 1.25, size=graph_count), -3.0, 3.0)
            candidate[graph] = (
                left + fraction[:, None] * route + jitter[:, None] * perpendicular
            )
            kind[graph] = 2

        in_bounds = np.all((candidate >= lower) & (candidate <= upper), axis=1)
        if site_value == 0:
            in_bounds &= candidate[:, 1] < boundary
        else:
            in_bounds &= candidate[:, 1] >= boundary
        distance, _ = anchor_tree.query(candidate, k=1)
        in_bounds &= distance >= 0.20

        for xy, sample_kind in zip(candidate[in_bounds], kind[in_bounds], strict=True):
            key = tuple(np.rint(xy / 0.05).astype(np.int64).tolist())
            if key in seen:
                continue
            seen.add(key)
            output.append(np.asarray((xy[0], xy[1], 1.5), dtype=np.float32))
            kinds.append(int(sample_kind))
            if len(output) == count:
                break

    return np.stack(output), np.asarray(kinds, dtype=np.uint8)


def prepare_positions(
    root: Path, point_count: int, sampling_seed: int, force: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    position_path = root / "positions.npy"
    site_path = root / "site.npy"
    kind_path = root / "sample_kind.npy"
    if force:
        for path in (position_path, site_path, kind_path, root / "state.json"):
            path.unlink(missing_ok=True)
    if all(path.is_file() for path in (position_path, site_path, kind_path)):
        positions = np.load(position_path)
        site = np.load(site_path)
        kind = np.load(kind_path)
        if len(positions) != point_count:
            raise ValueError(
                f"cached position count {len(positions)} != requested {point_count}"
            )
        return positions, site, kind, {
            "status": "reused",
            "digest": digest_arrays(positions, site, kind),
        }

    _, train_positions, test_positions, train_site, test_site = load_positions()
    boundary = _site_boundary(train_positions)
    counts = (point_count // 2, point_count - point_count // 2)
    position_parts: list[np.ndarray] = []
    site_parts: list[np.ndarray] = []
    kind_parts: list[np.ndarray] = []
    for site_value, count in enumerate(counts):
        rng = np.random.default_rng(sampling_seed + 100_003 * site_value)
        local_positions, local_kind = _sample_site_positions(
            site_value=site_value,
            count=count,
            train_positions=train_positions,
            test_positions=test_positions,
            train_site=train_site,
            test_site=test_site,
            boundary=boundary,
            rng=rng,
        )
        position_parts.append(local_positions)
        site_parts.append(np.full(count, site_value, dtype=np.int8))
        kind_parts.append(local_kind)
    positions = np.concatenate(position_parts, axis=0)
    site = np.concatenate(site_parts, axis=0)
    kind = np.concatenate(kind_parts, axis=0)
    np.save(position_path, positions)
    np.save(site_path, site)
    np.save(kind_path, kind)
    return positions, site, kind, {
        "status": "created",
        "digest": digest_arrays(positions, site, kind),
        "site_counts": np.bincount(site, minlength=2).astype(int).tolist(),
        "sample_kind_counts": np.bincount(kind, minlength=3).astype(int).tolist(),
        "site_boundary_y": boundary,
    }


def path_coordinates(
    tau: np.ndarray,
    theta_t: np.ndarray,
    phi_t: np.ndarray,
    site_value: int,
    configuration: Any,
) -> np.ndarray:
    yaw = math.radians(configuration.site_yaw_deg[site_value])
    tilt = math.radians(configuration.site_tilt_deg[site_value])
    horizontal = (
        MH
        * configuration.tx_horizontal_spacing
        * np.sin(theta_t)
        * np.sin(phi_t - yaw)
    ) % MH
    vertical = (
        MV
        * configuration.tx_vertical_spacing
        * (
            math.sin(tilt) * np.sin(theta_t) * np.cos(phi_t - yaw)
            - math.cos(tilt) * np.cos(theta_t)
        )
        + 2.0
    ) % MV
    delay = (tau * 30_000.0 * S_SC) % S_SC
    return np.stack((horizontal, vertical, delay), axis=-1).astype(np.float32)


def spectral_kernels(
    coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizontal_wave = np.exp(
        2j
        * np.pi
        * coordinate[..., 0, None]
        * np.arange(MH, dtype=np.float32)
        / float(MH)
    )
    vertical_wave = np.exp(
        2j
        * np.pi
        * coordinate[..., 1, None]
        * np.arange(MV, dtype=np.float32)
        / float(MV)
    )
    delay_wave = np.exp(
        -2j
        * np.pi
        * coordinate[..., 2, None]
        * np.arange(S_SC, dtype=np.float32)
        / float(S_SC)
    )
    horizontal = np.abs(np.fft.fft(horizontal_wave, axis=-1)) ** 2
    vertical = np.abs(np.fft.fft(vertical_wave, axis=-1)) ** 2
    delay = np.abs(np.fft.ifft(delay_wave, axis=-1)) ** 2
    horizontal /= np.maximum(horizontal.sum(axis=-1, keepdims=True), TINY)
    vertical /= np.maximum(vertical.sum(axis=-1, keepdims=True), TINY)
    delay /= np.maximum(delay.sum(axis=-1, keepdims=True), TINY)
    return (
        horizontal.astype(np.float32),
        vertical.astype(np.float32),
        delay.astype(np.float32),
    )


def render_sufficient_spectra(
    coordinate: np.ndarray,
    mass: np.ndarray,
    port: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    horizontal, vertical, delay = spectral_kernels(coordinate)
    energy = mass[..., None, None] * port
    pas = np.einsum(
        "bkpu,bkh,bkv->bphvu", energy, horizontal, vertical, optimize=True
    )
    pas /= np.maximum(pas.sum(axis=(1, 2, 3), keepdims=True), TINY)
    pdp = np.einsum("bkpu,bkd->bpud", energy, delay, optimize=True)
    pdp /= np.maximum(pdp.sum(axis=3, keepdims=True), TINY)
    return pas.astype(np.float32), pdp.astype(np.float32)


def compact_path_batch(
    traced: Any,
    batch_rows: np.ndarray,
    site_value: int,
    path_slots: int,
    configuration: Any,
) -> dict[str, np.ndarray]:
    packed = path_batch(
        traced,
        np.arange(len(batch_rows), dtype=np.int64),
        batch_rows,
        "reference",
    )
    batch_count = len(batch_rows)
    coordinate = np.zeros((batch_count, path_slots, 3), dtype=np.float32)
    mass = np.zeros((batch_count, path_slots), dtype=np.float32)
    port = np.zeros((batch_count, path_slots, MP, N_UE), dtype=np.float32)
    mask = np.zeros((batch_count, path_slots), dtype=bool)
    depth = np.zeros((batch_count, path_slots), dtype=np.uint8)
    path_count = np.zeros(batch_count, dtype=np.int32)

    offsets = packed["offsets"]
    for local in range(batch_count):
        start, end = int(offsets[local]), int(offsets[local + 1])
        path_count[local] = end - start
        if end <= start:
            continue
        local_fraction = np.asarray(
            packed["path_energy_fraction"][start:end], dtype=np.float32
        )
        take = min(path_slots, len(local_fraction))
        selected = np.argpartition(local_fraction, len(local_fraction) - take)[-take:]
        selected = selected[np.argsort(local_fraction[selected])[::-1]] + start
        local_mass = np.asarray(
            packed["path_energy_fraction"][selected], dtype=np.float32
        )
        local_mass /= np.maximum(local_mass.sum(dtype=np.float64), TINY)
        local_port = np.asarray(packed["port_energy"][selected], dtype=np.float32)
        local_port /= np.maximum(
            local_port.sum(axis=(1, 2), keepdims=True), TINY
        )
        local_coordinate = path_coordinates(
            np.asarray(packed["tau"][selected], dtype=np.float32),
            np.asarray(packed["theta_t"][selected], dtype=np.float32),
            np.asarray(packed["phi_t"][selected], dtype=np.float32),
            site_value,
            configuration,
        )
        coordinate[local, :take] = local_coordinate
        mass[local, :take] = local_mass
        port[local, :take] = local_port
        mask[local, :take] = True
        depth[local, :take] = np.asarray(
            packed["interaction_count"][selected], dtype=np.uint8
        )

    pas, pdp = render_sufficient_spectra(coordinate, mass, port)
    return {
        "coordinate": coordinate,
        "mass": mass,
        "port": port,
        "mask": mask,
        "depth": depth,
        "path_count": path_count,
        "pas": pas,
        "pdp": pdp,
    }


def seed_paths(root: Path, seed: int) -> dict[str, Path]:
    prefix = root / f"seed_{seed}"
    return {
        "coordinate": prefix.with_name(prefix.name + "_coordinate_f16.npy"),
        "mass": prefix.with_name(prefix.name + "_mass_f16.npy"),
        "port": prefix.with_name(prefix.name + "_port_f16.npy"),
        "mask": prefix.with_name(prefix.name + "_mask.npy"),
        "depth": prefix.with_name(prefix.name + "_depth.npy"),
        "path_count": prefix.with_name(prefix.name + "_path_count.npy"),
        "pas": prefix.with_name(prefix.name + "_pas_f16.npy"),
        "pdp": prefix.with_name(prefix.name + "_pdp_f16.npy"),
        "done": prefix.with_name(prefix.name + "_done.npy"),
    }


def generate_seed(
    *,
    root: Path,
    seed: int,
    positions: np.ndarray,
    site: np.ndarray,
    configuration: Any,
    runtime: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    paths = seed_paths(root, seed)
    count = len(positions)
    stores = {
        "coordinate": open_memmap(
            paths["coordinate"], np.float16, (count, args.path_slots, 3)
        ),
        "mass": open_memmap(paths["mass"], np.float16, (count, args.path_slots)),
        "port": open_memmap(
            paths["port"], np.float16, (count, args.path_slots, MP, N_UE)
        ),
        "mask": open_memmap(paths["mask"], np.bool_, (count, args.path_slots)),
        "depth": open_memmap(paths["depth"], np.uint8, (count, args.path_slots)),
        "path_count": open_memmap(paths["path_count"], np.int32, (count,)),
        "pas": open_memmap(paths["pas"], np.float16, (count, MP, MH, MV, N_UE)),
        "pdp": open_memmap(paths["pdp"], np.float16, (count, MP, N_UE, S_SC)),
        "done": open_memmap(paths["done"], np.bool_, (count,)),
    }
    if np.all(stores["done"]):
        return {
            "seed": int(seed),
            "status": "reused",
            "rows": int(count),
            "outputs": {name: PATHS.relative(path) for name, path in paths.items()},
        }

    setup, _, _, _, _ = load_positions()
    started = time.time()
    material_values: dict[str, float] = {}
    for site_value in np.unique(site):
        scene = runtime["load_scene"](
            str(SCENE_XML), merge_shapes=True, remove_duplicate_vertices=True
        )
        scene.frequency = configuration.frequency_hz
        scene.tx_array = runtime["PlanarArray"](
            num_rows=8,
            num_cols=16,
            vertical_spacing=configuration.tx_vertical_spacing,
            horizontal_spacing=configuration.tx_horizontal_spacing,
            pattern="tr38901",
            polarization="VH",
        )
        scene.rx_array = runtime["PlanarArray"](
            num_rows=2,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="cross",
        )
        material_values = configure_materials(
            scene, configuration.scattering_coefficient
        )
        bs_position = setup.bs_positions[int(site_value)]
        transmitter = runtime["Transmitter"](
            name=f"e88_tx_site{int(site_value)}_seed{seed}",
            position=runtime["mi"].Point3f(*map(float, bs_position)),
            look_at=tx_target(
                bs_position,
                configuration.site_yaw_deg[int(site_value)],
                configuration.site_tilt_deg[int(site_value)],
                runtime["mi"],
            ),
            power_dbm=44.0,
        )
        scene.add(transmitter)
        solver = runtime["PathSolver"]()
        receiver_names: list[str] = []
        local_rows = np.flatnonzero(site == site_value)
        for offset in range(0, len(local_rows), args.batch_size):
            batch_rows = local_rows[offset : offset + args.batch_size]
            if np.all(stores["done"][batch_rows]):
                continue
            for name in receiver_names:
                scene.remove(name)
            receiver_names = []
            for row in batch_rows:
                name = f"e88_rx_{int(row):06d}"
                scene.add(
                    runtime["Receiver"](
                        name=name,
                        position=runtime["mi"].Point3f(
                            *map(float, positions[row])
                        ),
                        look_at=transmitter,
                    )
                )
                receiver_names.append(name)
            select_drjit_cuda_device(runtime["dr"], args.cuda_device)
            traced = solver(
                scene,
                max_depth=configuration.max_depth,
                max_num_paths_per_src=configuration.max_num_paths_per_src,
                samples_per_src=configuration.samples_per_src,
                synthetic_array=configuration.synthetic_array,
                los=True,
                specular_reflection=True,
                diffuse_reflection=configuration.diffuse_reflection,
                refraction=True,
                diffraction=configuration.diffraction,
                edge_diffraction=configuration.edge_diffraction,
                diffraction_lit_region=True,
                seed=int(seed + 100_000 * int(site_value) + int(batch_rows[0])),
            )
            compact = compact_path_batch(
                traced,
                batch_rows,
                int(site_value),
                args.path_slots,
                configuration,
            )
            for name, value in compact.items():
                stores[name][batch_rows] = value.astype(stores[name].dtype)
            stores["done"][batch_rows] = True
            for value in stores.values():
                value.flush()
            del traced, compact
            runtime["dr"].sync_thread()
            completed = int(np.sum(stores["done"]))
            if completed == count or completed % 512 < len(batch_rows):
                print(
                    f"E88 seed={seed} site={int(site_value)} "
                    f"{completed}/{count} elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
        for name in receiver_names:
            scene.remove(name)
        del scene, solver
        runtime["dr"].sync_thread()

    if not np.all(stores["done"]):
        raise RuntimeError(f"seed {seed} did not complete")
    return {
        "seed": int(seed),
        "status": "generated",
        "rows": int(count),
        "seconds": float(time.time() - started),
        "zero_path_fraction": float(np.mean(np.asarray(stores["path_count"]) == 0)),
        "mean_path_count": float(np.mean(np.asarray(stores["path_count"]))),
        "materials": material_values,
        "outputs": {name: PATHS.relative(path) for name, path in paths.items()},
    }


def main() -> None:
    args = parse_args()
    if args.point_count < 2:
        raise ValueError("--point-count must be at least 2")
    if not 1 <= args.seed_count <= len(SEEDS):
        raise ValueError(f"--seed-count must be in [1, {len(SEEDS)}]")
    if args.path_slots <= 0 or args.batch_size <= 0:
        raise ValueError("--path-slots and --batch-size must be positive")

    configuration = CONFIGS[0]
    root = ARTIFACT_DIR / args.selection_tag
    root.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    positions, site, kind, sampling = prepare_positions(
        root, args.point_count, args.sampling_seed, args.force_sampling
    )
    expected = {
        "experiment": VERSION,
        "configuration": asdict(configuration),
        "selection_tag": args.selection_tag,
        "point_count": int(args.point_count),
        "path_slots": int(args.path_slots),
        "sampling_seed": int(args.sampling_seed),
        "position_digest": digest_arrays(positions, site, kind),
        "seed_order": list(SEEDS),
    }
    state_path = root / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"E88 state mismatch for {key}")
    else:
        state = {**expected, "completed_seeds": []}
        atomic_json(state_path, state)

    prepare_scene()
    runtime = prepare_runtime(args)
    runs = []
    for seed in SEEDS[: args.seed_count]:
        run = generate_seed(
            root=root,
            seed=int(seed),
            positions=positions,
            site=site,
            configuration=configuration,
            runtime=runtime,
            args=args,
        )
        runs.append(run)
        if seed not in state["completed_seeds"]:
            state["completed_seeds"].append(int(seed))
            atomic_json(state_path, state)

    report = {
        "experiment": VERSION,
        "status": "teacher_generation_complete",
        "primary_hypothesis": (
            "Dense Sionna labels at new map positions add the missing spatial coverage "
            "needed for a map-only student to learn path birth, death and energy changes."
        ),
        "configuration": asdict(configuration),
        "arguments": vars(args),
        "sampling": sampling,
        "position_digest": expected["position_digest"],
        "environment": {
            "sionna": runtime["sionna"].__version__,
            "mitsuba_variant": runtime["mi"].variant(),
            "single_visible_gpu_required": True,
        },
        "runs": runs,
        "output_root": PATHS.relative(root),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
