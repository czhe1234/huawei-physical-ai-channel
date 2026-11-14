#!/usr/bin/env python3
"""v251: trace geometry-only paths on an official-point-cloud scene.

The scene mesh is reconstructed from ``Round1_Map.ply`` by the existing
official point-cloud wall-patch builder.  Sionna is used only as a visibility
and interaction solver.  This script never calls ``Paths.cfr()`` and never
stores path coefficients or channel labels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from project_paths import PATHS


VERSION_NUMBER = 251
VERSION = "v251_pointcloud_sionna_paths"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
SCENE_DIR = OUT_DIR / "scene"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = OUT_DIR / f"{VERSION}.npz"

SOURCE_OBJ = PATHS.shared_features / "official_pointcloud_wall_patches.obj"
BS_POS = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
LIGHT_SPEED = 299_792_458.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=2500)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--samples-per-src", type=int, default=120_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=25101)
    parser.add_argument("--frequency", type=float, default=6.0e9)
    parser.add_argument("--subcarrier-spacing", type=float, default=30e3)
    parser.add_argument("--mitsuba-variant", default="llvm_ad_mono_polarized")
    parser.add_argument(
        "--diffuse-reflection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--refraction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--diffraction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--edge-diffraction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--depth-balanced-ranking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def prepare_scene_files() -> Path:
    if not SOURCE_OBJ.is_file():
        raise FileNotFoundError(SOURCE_OBJ)
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    wall_obj = SCENE_DIR / "official_walls.obj"
    ground_obj = SCENE_DIR / "ground.obj"
    scene_xml = SCENE_DIR / "scene.xml"

    # Lower the reconstructed walls to z=0 so visibility rays cannot pass
    # through the small sampling gap below the original point-cloud patches.
    lines = []
    for line in SOURCE_OBJ.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            fields = line.split()
            x, y, z = map(float, fields[1:4])
            if z < 2.1:
                z = 0.0
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
        else:
            lines.append(line)
    wall_obj.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ground_obj.write_text(
        "\n".join(
            [
                "# ground reconstructed for the official point-cloud scene",
                "v -500.0 -420.0 0.0",
                "v 500.0 -420.0 0.0",
                "v 500.0 420.0 0.0",
                "v -500.0 420.0 0.0",
                "f 1 2 3 4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    scene_xml.write_text(
        "\n".join(
            [
                '<scene version="2.1.0">',
                '  <bsdf type="itu-radio-material" id="concrete">',
                '    <string name="type" value="concrete"/>',
                '    <float name="thickness" value="0.1"/>',
                "  </bsdf>",
                '  <shape type="obj" id="mesh-official-walls">',
                '    <string name="filename" value="official_walls.obj"/>',
                '    <boolean name="face_normals" value="true"/>',
                '    <ref id="concrete" name="bsdf"/>',
                "  </shape>",
                '  <shape type="obj" id="mesh-ground">',
                '    <string name="filename" value="ground.obj"/>',
                '    <boolean name="face_normals" value="true"/>',
                '    <ref id="concrete" name="bsdf"/>',
                "  </shape>",
                "</scene>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scene_xml


def allocate(count: int, max_paths: int, max_depth: int) -> dict[str, np.ndarray]:
    return {
        "valid": np.zeros((count, max_paths), dtype=bool),
        "tau": np.full((count, max_paths), np.nan, dtype=np.float32),
        "path_length": np.full((count, max_paths), np.nan, dtype=np.float32),
        "delay_bin_physical": np.full((count, max_paths), np.nan, dtype=np.float32),
        "theta_t": np.full((count, max_paths), np.nan, dtype=np.float32),
        "phi_t": np.full((count, max_paths), np.nan, dtype=np.float32),
        "theta_r": np.full((count, max_paths), np.nan, dtype=np.float32),
        "phi_r": np.full((count, max_paths), np.nan, dtype=np.float32),
        "interaction_count": np.zeros((count, max_paths), dtype=np.int8),
        "interaction_mask": np.zeros((count, max_paths), dtype=np.int16),
        "interactions": np.zeros((count, max_paths, max_depth), dtype=np.int8),
        "vertices": np.full((count, max_paths, max_depth, 3), np.nan, dtype=np.float32),
        "objects": np.full((count, max_paths, max_depth), -1, dtype=np.int32),
        "score": np.full((count, max_paths), -np.inf, dtype=np.float32),
    }


def remove_receivers(scene, names: list[str]) -> None:
    for name in names:
        if scene.get(name) is not None:
            scene.remove(name)


def select_paths(
    paths,
    max_paths: int,
    max_depth: int,
    subcarrier_spacing: float,
    depth_balanced_ranking: bool = False,
    num_subcarriers: int = 192,
) -> list[dict[str, np.ndarray]]:
    valid = paths.valid.numpy()[:, 0].astype(bool)
    tau = paths.tau.numpy()[:, 0].astype(np.float32)
    theta_t = paths.theta_t.numpy()[:, 0].astype(np.float32)
    phi_t = paths.phi_t.numpy()[:, 0].astype(np.float32)
    theta_r = paths.theta_r.numpy()[:, 0].astype(np.float32)
    phi_r = paths.phi_r.numpy()[:, 0].astype(np.float32)
    interactions = paths.interactions.numpy()[:, :, 0].transpose(1, 2, 0).astype(np.int8)
    vertices = paths.vertices.numpy()[:, :, 0].transpose(1, 2, 0, 3).astype(np.float32)
    objects = paths.objects.numpy()[:, :, 0].transpose(1, 2, 0).astype(np.int64)
    objects = np.where(objects >= np.iinfo(np.uint32).max, -1, objects).astype(np.int32)
    output = []
    for row in range(len(valid)):
        ids = np.flatnonzero(valid[row] & np.isfinite(tau[row]) & (tau[row] >= 0.0))
        if len(ids) == 0:
            output.append({"ids": np.empty(0, dtype=np.int64)})
            continue
        inter = interactions[row, ids, :max_depth]
        depth = np.sum(inter != 0, axis=1).astype(np.int8)
        mask = np.bitwise_or.reduce(inter.astype(np.int16), axis=1)
        length = tau[row, ids].astype(np.float64) * LIGHT_SPEED
        direct = depth == 0
        diffracted = (mask & 8) != 0
        # Geometry-only ranking: preserve direct paths, then short low-order
        # specular/diffracted paths. No path coefficient enters this score.
        rank_score = (
            3.0 * direct.astype(np.float64)
            + 0.35 * diffracted.astype(np.float64)
            - 0.42 * depth.astype(np.float64)
            - 0.008 * np.maximum(length - np.min(length), 0.0)
        )
        ranked = np.argsort(rank_score)[::-1]
        if depth_balanced_ranking:
            quota = max(int(np.ceil(max_paths / max(max_depth + 1, 1))), 1)
            selected_parts = [
                ranked[depth[ranked] == value][:quota]
                for value in range(max_depth + 1)
            ]
            nonempty = [value for value in selected_parts if len(value)]
            selected = (
                np.concatenate(nonempty)
                if nonempty
                else np.empty(0, dtype=np.int64)
            )
            if len(selected) < max_paths:
                remaining = ranked[~np.isin(ranked, selected)]
                selected = np.concatenate(
                    [selected, remaining[: max_paths - len(selected)]]
                )
            order = selected[:max_paths]
        else:
            order = ranked[:max_paths]
        ids = ids[order]
        depth = depth[order]
        mask = mask[order]
        length = length[order]
        rank_score = rank_score[order]
        output.append(
            {
                "ids": ids,
                "tau": tau[row, ids],
                "path_length": length.astype(np.float32),
                "delay_bin_physical": (
                    tau[row, ids] * float(subcarrier_spacing) * float(num_subcarriers)
                ).astype(np.float32),
                "theta_t": theta_t[row, ids],
                "phi_t": phi_t[row, ids],
                "theta_r": theta_r[row, ids],
                "phi_r": phi_r[row, ids],
                "interaction_count": depth,
                "interaction_mask": mask,
                "interactions": interactions[row, ids, :max_depth],
                "vertices": vertices[row, ids, :max_depth],
                "objects": objects[row, ids, :max_depth],
                "score": rank_score.astype(np.float32),
            }
        )
    return output


def trace_split(
    *,
    prefix: str,
    positions: np.ndarray,
    args: argparse.Namespace,
    scene,
    tx,
    mi,
    Receiver,
    PathSolver,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays = allocate(len(positions), args.max_paths, args.max_depth)
    names: list[str] = []
    solver = PathSolver()
    batch_reports = []
    started = time.time()
    for start in range(0, len(positions), args.batch_size):
        end = min(start + args.batch_size, len(positions))
        remove_receivers(scene, names)
        names = []
        for local, pos in enumerate(positions[start:end]):
            name = f"rx_{prefix}_{start + local:05d}"
            scene.add(
                Receiver(
                    name=name,
                    position=mi.Point3f(*map(float, pos)),
                    look_at=tx,
                )
            )
            names.append(name)
        t0 = time.time()
        paths = solver(
            scene,
            max_depth=args.max_depth,
            max_num_paths_per_src=args.max_num_paths_per_src,
            samples_per_src=args.samples_per_src,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=args.diffuse_reflection,
            refraction=args.refraction,
            diffraction=args.diffraction,
            edge_diffraction=args.edge_diffraction,
            diffraction_lit_region=True,
            seed=args.seed + (1_000_000 if prefix == "test" else 0) + start,
        )
        selected = select_paths(
            paths,
            args.max_paths,
            args.max_depth,
            args.subcarrier_spacing,
            args.depth_balanced_ranking,
        )
        counts = []
        for local, item in enumerate(selected):
            count = len(item["ids"])
            counts.append(count)
            if count == 0:
                continue
            row = start + local
            arrays["valid"][row, :count] = True
            for key in arrays:
                if key == "valid":
                    continue
                arrays[key][row, :count] = item[key]
        seconds = time.time() - t0
        batch_reports.append(
            {
                "start": int(start),
                "end": int(end),
                "seconds": float(seconds),
                "path_count_mean": float(np.mean(counts)),
                "path_count_max": int(max(counts, default=0)),
                "zero_path_count": int(np.sum(np.asarray(counts) == 0)),
            }
        )
        print(
            f"v251 {prefix} {end}/{len(positions)} seconds={seconds:.3f} "
            f"paths_mean={np.mean(counts):.2f} zero={np.sum(np.asarray(counts) == 0)}",
            flush=True,
        )
    remove_receivers(scene, names)
    count = arrays["valid"].sum(axis=1)
    return arrays, {
        "count": int(len(positions)),
        "seconds": float(time.time() - started),
        "path_count": {
            "mean": float(np.mean(count)),
            "median": float(np.median(count)),
            "p90": float(np.quantile(count, 0.9)),
            "max": int(np.max(count, initial=0)),
            "zero": int(np.sum(count == 0)),
        },
        "batch_reports": batch_reports,
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scene_xml = prepare_scene_files()

    import mitsuba as mi

    mi.set_variant(args.mitsuba_variant)
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    train_pos_all = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos_all = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    train_count = min(max(args.train_count, 0), len(train_pos_all))
    test_count = min(max(args.test_count, 0), len(test_pos_all))
    train_pos = train_pos_all[:train_count]
    test_pos = test_pos_all[:test_count]

    print(f"v251 mitsuba_variant={mi.variant()} scene={scene_xml}", flush=True)
    scene = load_scene(str(scene_xml), merge_shapes=True)
    scene.frequency = args.frequency
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    tx = Transmitter(
        name="tx",
        position=mi.Point3f(*map(float, BS_POS)),
        look_at=mi.Point3f(0.0, 0.0, 1.5),
    )
    scene.add(tx)

    train, train_report = trace_split(
        prefix="train",
        positions=train_pos,
        args=args,
        scene=scene,
        tx=tx,
        mi=mi,
        Receiver=Receiver,
        PathSolver=PathSolver,
    )
    test, test_report = trace_split(
        prefix="test",
        positions=test_pos,
        args=args,
        scene=scene,
        tx=tx,
        mi=mi,
        Receiver=Receiver,
        PathSolver=PathSolver,
    )
    payload: dict[str, np.ndarray] = {
        "train_indices": np.arange(train_count, dtype=np.int32),
        "test_indices": np.arange(test_count, dtype=np.int32),
        "train_pos": train_pos,
        "test_pos": test_pos,
        "bs_pos": BS_POS,
    }
    for prefix, arrays in (("train", train), ("test", test)):
        for key, value in arrays.items():
            payload[f"{prefix}_{key}"] = value
    np.savez_compressed(OUTPUT, **payload)

    report = {
        "method": VERSION,
        "status": "geometry_cache_only",
        "primary_hypothesis": (
            "Sionna visibility solving on the official-point-cloud mesh can replace "
            "the invalid compressed 15-path candidate set without using RT channels."
        ),
        "compliance_boundary": {
            "uses": [
                "official Round1 point-cloud-derived wall mesh",
                "official train/test coordinates",
                "Sionna PathSolver geometry and visibility",
            ],
            "does_not_use": [
                "Paths.cfr or any ray-traced H",
                "path complex coefficients",
                "synthetic/Sionna channel labels",
                "v212/v217 teacher outputs",
                "local replay truth",
                "hidden online labels",
            ],
        },
        "config": vars(args),
        "mitsuba_variant": mi.variant(),
        "scene": {
            "source_obj": PATHS.relative(SOURCE_OBJ),
            "generated_xml": PATHS.relative(scene_xml),
        },
        "train": train_report,
        "test": test_report,
        "outputs": {
            "paths": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
