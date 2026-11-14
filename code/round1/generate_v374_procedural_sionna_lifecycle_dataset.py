#!/usr/bin/env python3
"""Generate independent procedural Sionna scenes for path lifecycle pretraining."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from project_paths import PATHS


VERSION_NUMBER = 374
VERSION = "v374_procedural_sionna_lifecycle_dataset"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
SCENE_ROOT = OUT_DIR / "scenes"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"

LIGHT_SPEED = 299_792_458.0
MATERIALS = ("concrete", "brick", "glass", "wood")


@dataclass
class SceneSpec:
    boxes: np.ndarray
    material: np.ndarray
    bs: np.ndarray
    positions: np.ndarray
    frequency: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-count", type=int, default=24)
    parser.add_argument("--points-per-scene", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-buildings", type=int, default=14)
    parser.add_argument("--max-buildings", type=int, default=24)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--samples-per-src", type=int, default=160_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=80_000)
    parser.add_argument("--min-family-points", type=int, default=4)
    parser.add_argument("--seed", type=int, default=37401)
    parser.add_argument(
        "--mitsuba-variant", default="llvm_ad_mono_polarized"
    )
    return parser.parse_args()


def overlaps(box: np.ndarray, boxes: list[np.ndarray], margin: float) -> bool:
    for old in boxes:
        if (
            abs(float(box[0] - old[0]))
            < 0.5 * float(box[2] + old[2]) + margin
            and abs(float(box[1] - old[1]))
            < 0.5 * float(box[3] + old[3]) + margin
        ):
            return True
    return False


def point_inside(point: np.ndarray, boxes: np.ndarray, margin: float) -> bool:
    return bool(
        np.any(
            (np.abs(point[0] - boxes[:, 0]) <= 0.5 * boxes[:, 2] + margin)
            & (np.abs(point[1] - boxes[:, 1]) <= 0.5 * boxes[:, 3] + margin)
        )
    )


def random_scene_spec(
    rng: np.random.Generator,
    points: int,
    min_buildings: int,
    max_buildings: int,
) -> SceneSpec:
    target_buildings = int(rng.integers(min_buildings, max_buildings + 1))
    boxes: list[np.ndarray] = []
    attempts = 0
    while len(boxes) < target_buildings and attempts < 5000:
        attempts += 1
        box = np.asarray(
            [
                rng.uniform(-105.0, 105.0),
                rng.uniform(-105.0, 105.0),
                rng.uniform(12.0, 34.0),
                rng.uniform(12.0, 34.0),
                rng.uniform(8.0, 36.0),
            ],
            dtype=np.float32,
        )
        if overlaps(box, boxes, margin=5.0):
            continue
        boxes.append(box)
    if len(boxes) < min_buildings:
        raise RuntimeError("failed to generate enough non-overlapping buildings")
    box_array = np.stack(boxes)
    material = rng.integers(0, len(MATERIALS), size=len(boxes), dtype=np.int8)

    free_candidates = []
    while len(free_candidates) < points + 16:
        point = rng.uniform(-118.0, 118.0, size=2).astype(np.float32)
        if not point_inside(point, box_array, margin=1.5):
            free_candidates.append(point)
    free = np.stack(free_candidates)
    bs_xy = free[0]
    bs = np.asarray(
        [bs_xy[0], bs_xy[1], rng.uniform(20.0, 34.0)], dtype=np.float32
    )
    positions = np.column_stack(
        [free[1 : points + 1], np.full(points, 1.5, dtype=np.float32)]
    ).astype(np.float32)
    frequency = float(rng.choice(np.asarray([3.5e9, 6.0e9, 6.7e9])))
    return SceneSpec(box_array, material, bs, positions, frequency)


def box_obj(box: np.ndarray) -> str:
    cx, cy, width, depth, height = map(float, box)
    x0, x1 = cx - width / 2.0, cx + width / 2.0
    y0, y1 = cy - depth / 2.0, cy + depth / 2.0
    vertices = [
        (x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0),
        (x0, y0, height), (x1, y0, height), (x1, y1, height), (x0, y1, height),
    ]
    faces = [
        (1, 4, 3), (1, 3, 2),
        (5, 6, 7), (5, 7, 8),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 4, 8), (3, 8, 7),
        (4, 1, 5), (4, 5, 8),
    ]
    rows = ["# procedural rectangular building"]
    rows.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    rows.extend(f"f {a} {b} {c}" for a, b, c in faces)
    return "\n".join(rows) + "\n"


def ground_obj() -> str:
    return "\n".join(
        [
            "# procedural ground",
            "v -150.0 -150.0 0.0",
            "v 150.0 -150.0 0.0",
            "v 150.0 150.0 0.0",
            "v -150.0 150.0 0.0",
            "f 1 2 3",
            "f 1 3 4",
            "",
        ]
    )


def write_scene(scene_dir: Path, spec: SceneSpec) -> Path:
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "ground.obj").write_text(ground_obj(), encoding="utf-8")
    for index, box in enumerate(spec.boxes):
        (scene_dir / f"building_{index:03d}.obj").write_text(
            box_obj(box), encoding="utf-8"
        )
    rows = ['<scene version="2.1.0">']
    for material in MATERIALS:
        rows.extend(
            [
                f'  <bsdf type="itu-radio-material" id="{material}">',
                f'    <string name="type" value="{material}"/>',
                '    <float name="thickness" value="0.1"/>',
                "  </bsdf>",
            ]
        )
    rows.extend(
        [
            '  <shape type="obj" id="mesh-ground">',
            '    <string name="filename" value="ground.obj"/>',
            '    <boolean name="face_normals" value="true"/>',
            '    <ref id="concrete" name="bsdf"/>',
            "  </shape>",
        ]
    )
    for index, material_id in enumerate(spec.material):
        material = MATERIALS[int(material_id)]
        rows.extend(
            [
                f'  <shape type="obj" id="mesh-building-{index:03d}">',
                f'    <string name="filename" value="building_{index:03d}.obj"/>',
                '    <boolean name="face_normals" value="true"/>',
                f'    <ref id="{material}" name="bsdf"/>',
                "  </shape>",
            ]
        )
    rows.extend(["</scene>", ""])
    path = scene_dir / "scene.xml"
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def remove_receivers(scene, names: list[str]) -> None:
    for name in names:
        if scene.get(name) is not None:
            scene.remove(name)


def path_key(
    interactions: np.ndarray,
    objects: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    active = np.flatnonzero(interactions != 0)
    if len(active) == 0:
        return tuple()
    return tuple(
        (int(interactions[index]), int(objects[index])) for index in active
    )


def trace_scene(
    *,
    scene_id: int,
    scene_xml: Path,
    spec: SceneSpec,
    args: argparse.Namespace,
    mi,
    load_scene,
    Transmitter,
    Receiver,
    PlanarArray,
    PathSolver,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    scene = load_scene(str(scene_xml), merge_shapes=False)
    scene.frequency = spec.frequency
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
        position=mi.Point3f(*map(float, spec.bs)),
        look_at=mi.Point3f(0.0, 0.0, 1.5),
        power_dbm=30.0,
    )
    scene.add(tx)
    solver = PathSolver()
    family: dict[tuple[tuple[int, int], ...], dict[str, np.ndarray]] = {}
    receiver_names: list[str] = []
    batch_reports = []
    started = time.time()

    for start in range(0, len(spec.positions), args.batch_size):
        end = min(start + args.batch_size, len(spec.positions))
        remove_receivers(scene, receiver_names)
        receiver_names = []
        for index, position in enumerate(spec.positions[start:end], start=start):
            name = f"rx_{index:05d}"
            scene.add(
                Receiver(
                    name=name,
                    position=mi.Point3f(*map(float, position)),
                    look_at=tx,
                )
            )
            receiver_names.append(name)
        t0 = time.time()
        paths = solver(
            scene,
            max_depth=args.max_depth,
            max_num_paths_per_src=args.max_num_paths_per_src,
            samples_per_src=args.samples_per_src,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            diffraction=True,
            edge_diffraction=False,
            diffraction_lit_region=True,
            seed=args.seed + 100_000 * scene_id + start,
        )
        solve_seconds = time.time() - t0
        valid = paths.valid.numpy()[:, 0].astype(bool)
        tau = paths.tau.numpy()[:, 0].astype(np.float64)
        theta_r = paths.theta_r.numpy()[:, 0].astype(np.float64)
        phi_r = paths.phi_r.numpy()[:, 0].astype(np.float64)
        interactions = paths.interactions.numpy()[:, :, 0].transpose(1, 2, 0)
        objects = paths.objects.numpy()[:, :, 0].transpose(1, 2, 0).astype(np.int64)
        objects = np.where(objects >= np.iinfo(np.uint32).max, -1, objects)
        coefficient, _ = paths.cir(normalize_delays=False, out_type="numpy")
        raw = coefficient[:, 0, 0, 0, :, 0]
        batch_path_count = 0
        for local in range(end - start):
            point_id = start + local
            ids = np.flatnonzero(valid[local] & np.isfinite(tau[local]) & (tau[local] >= 0.0))
            batch_path_count += len(ids)
            for path_id in ids:
                key = path_key(
                    interactions[local, path_id, : args.max_depth],
                    objects[local, path_id, : args.max_depth],
                )
                power = float(abs(raw[local, path_id]) ** 2)
                if not np.isfinite(power) or power <= 0.0:
                    continue
                if key not in family:
                    family[key] = {
                        "valid": np.zeros(len(spec.positions), dtype=bool),
                        "delay": np.full(len(spec.positions), np.nan, dtype=np.float32),
                        "arrival_xy": np.full((len(spec.positions), 2), np.nan, dtype=np.float32),
                        "power": np.zeros(len(spec.positions), dtype=np.float32),
                    }
                row = family[key]
                if row["valid"][point_id] and power <= float(row["power"][point_id]):
                    continue
                row["valid"][point_id] = True
                row["delay"][point_id] = float(tau[local, path_id])
                row["arrival_xy"][point_id] = np.asarray(
                    [
                        math.sin(float(theta_r[local, path_id]))
                        * math.cos(float(phi_r[local, path_id])),
                        math.sin(float(theta_r[local, path_id]))
                        * math.sin(float(phi_r[local, path_id])),
                    ],
                    dtype=np.float32,
                )
                row["power"][point_id] = power
        batch_reports.append(
            {
                "start": start,
                "end": end,
                "solve_seconds": solve_seconds,
                "raw_path_count": int(batch_path_count),
            }
        )
        print(
            f"v374 scene={scene_id} points={end}/{len(spec.positions)} "
            f"paths={batch_path_count} families={len(family)} solve={solve_seconds:.2f}s",
            flush=True,
        )
    remove_receivers(scene, receiver_names)

    retained = [
        (key, value)
        for key, value in family.items()
        if int(np.sum(value["valid"])) >= args.min_family_points
    ]
    retained.sort(
        key=lambda item: (int(np.sum(item[1]["valid"])), float(np.sum(item[1]["power"]))),
        reverse=True,
    )
    family_count = len(retained)
    interactions_out = np.zeros((family_count, args.max_depth), dtype=np.int8)
    objects_out = np.full((family_count, args.max_depth), -1, dtype=np.int32)
    valid_out = np.zeros((len(spec.positions), family_count), dtype=bool)
    delay_out = np.full((len(spec.positions), family_count), np.nan, dtype=np.float32)
    arrival_out = np.full((len(spec.positions), family_count, 2), np.nan, dtype=np.float32)
    power_out = np.zeros((len(spec.positions), family_count), dtype=np.float32)
    for family_id, (key, value) in enumerate(retained):
        for depth, (interaction, object_id) in enumerate(key[: args.max_depth]):
            interactions_out[family_id, depth] = interaction
            objects_out[family_id, depth] = object_id
        valid_out[:, family_id] = value["valid"]
        delay_out[:, family_id] = value["delay"]
        arrival_out[:, family_id] = value["arrival_xy"]
        power_out[:, family_id] = value["power"]
    total_power = np.sum(power_out, axis=1, dtype=np.float64)
    normalized_power = np.divide(
        power_out,
        np.maximum(total_power[:, None], 1e-30),
        out=np.zeros_like(power_out),
        where=total_power[:, None] > 0.0,
    )
    payload = {
        "positions": spec.positions,
        "bs": spec.bs,
        "boxes": spec.boxes,
        "material": spec.material,
        "frequency": np.asarray(spec.frequency, dtype=np.float64),
        "family_interactions": interactions_out,
        "family_objects": objects_out,
        "valid": valid_out,
        "delay_seconds": delay_out,
        "delay_bin": (delay_out * (192.0 * 30e3)).astype(np.float32),
        "arrival_xy": arrival_out,
        "power": power_out,
        "normalized_power": normalized_power.astype(np.float32),
    }
    report = {
        "scene_id": scene_id,
        "frequency": spec.frequency,
        "building_count": int(len(spec.boxes)),
        "point_count": int(len(spec.positions)),
        "raw_family_count": int(len(family)),
        "retained_family_count": family_count,
        "nonzero_point_count": int(np.sum(total_power > 0.0)),
        "mean_active_families": float(np.mean(np.sum(valid_out, axis=1))) if family_count else 0.0,
        "seconds": float(time.time() - started),
        "batches": batch_reports,
    }
    return payload, report


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCENE_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    import mitsuba as mi

    mi.set_variant(args.mitsuba_variant)
    from sionna.rt import (
        PathSolver,
        PlanarArray,
        Receiver,
        Transmitter,
        load_scene,
    )

    rng = np.random.default_rng(args.seed)
    scene_reports = []
    output_paths = []
    started = time.time()
    for scene_id in range(args.scene_count):
        scene_dir = SCENE_ROOT / f"scene_{scene_id:03d}"
        spec = random_scene_spec(
            rng,
            args.points_per_scene,
            args.min_buildings,
            args.max_buildings,
        )
        scene_xml = write_scene(scene_dir, spec)
        payload, scene_report = trace_scene(
            scene_id=scene_id,
            scene_xml=scene_xml,
            spec=spec,
            args=args,
            mi=mi,
            load_scene=load_scene,
            Transmitter=Transmitter,
            Receiver=Receiver,
            PlanarArray=PlanarArray,
            PathSolver=PathSolver,
        )
        output = scene_dir / "lifecycle.npz"
        np.savez_compressed(output, **payload)
        output_paths.append(PATHS.relative(output))
        scene_report["output"] = PATHS.relative(output)
        scene_reports.append(scene_report)

    report = {
        "method": VERSION,
        "status": "procedural_sionna_dataset_generated",
        "scope": "independent procedural scenes for generic lifecycle pretraining only",
        "compliance": {
            "uses": [
                "procedurally generated building geometry and independent UE coordinates",
                "Sionna PathSolver path identities, delays, arrival directions, and CIR power",
            ],
            "does_not_use": [
                "restored Etoile scene or official train/test coordinates",
                "official H, v212 teacher, hidden/online truth, or submission outputs",
            ],
        },
        "parameters": vars(args),
        "scene_reports": scene_reports,
        "summary": {
            "scene_count": len(scene_reports),
            "point_count": int(sum(row["point_count"] for row in scene_reports)),
            "retained_family_count": int(
                sum(row["retained_family_count"] for row in scene_reports)
            ),
            "seconds": float(time.time() - started),
        },
        "outputs": output_paths,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
