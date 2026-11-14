#!/usr/bin/env python3
"""Generate dense procedural Sionna cities matched to public geometry statistics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import generate_v374_procedural_sionna_lifecycle_dataset as v374
from project_paths import PATHS


VERSION_NUMBER = 381
VERSION = "v381_dense_urban_sionna_dataset"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
SCENE_ROOT = OUT_DIR / "scenes"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OFFICIAL_FEATURE_CACHE = PATHS.cache_dir(379) / "v379_official_lifecycle_features.npz"

FREQUENCY = 6.693e9
BS = np.asarray([-122.0, 28.0, 25.0], dtype=np.float32)
CITY_MIN = -132.0
CITY_MAX = 132.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-count", type=int, default=30)
    parser.add_argument("--points-per-scene", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--samples-per-src", type=int, default=200_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=100_000)
    parser.add_argument("--min-family-points", type=int, default=4)
    parser.add_argument("--seed", type=int, default=38101)
    parser.add_argument("--morphology-audit-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mitsuba-variant", default="llvm_ad_mono_polarized"
    )
    return parser.parse_args()


def random_grid_edges(
    rng: np.random.Generator,
    minimum: float,
    maximum: float,
    cells: int,
) -> np.ndarray:
    base = np.linspace(minimum, maximum, cells + 1)
    if cells > 1:
        step = (maximum - minimum) / cells
        base[1:-1] += rng.uniform(-0.15 * step, 0.15 * step, size=cells - 1)
    return base.astype(np.float32)


def append_rectangle(
    boxes: list[np.ndarray],
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    height: float,
) -> None:
    if x1 - x0 < 4.0 or y1 - y0 < 4.0:
        return
    boxes.append(
        np.asarray(
            [
                0.5 * (x0 + x1),
                0.5 * (y0 + y1),
                x1 - x0,
                y1 - y0,
                height,
            ],
            dtype=np.float32,
        )
    )


def block_buildings(
    rng: np.random.Generator,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> list[np.ndarray]:
    inset = float(rng.uniform(0.8, 2.5))
    x0 += inset
    x1 -= inset
    y0 += inset
    y1 -= inset
    if x1 - x0 < 8.0 or y1 - y0 < 8.0:
        return []
    height = float(rng.uniform(9.0, 36.0))
    style = rng.choice(
        np.asarray(["rectangle", "l_shape", "u_shape", "split"]),
        p=np.asarray([0.58, 0.24, 0.06, 0.12]),
    )
    boxes: list[np.ndarray] = []
    if style == "rectangle":
        append_rectangle(boxes, x0=x0, x1=x1, y0=y0, y1=y1, height=height)
    elif style == "l_shape":
        x_cut = float(x0 + rng.uniform(0.55, 0.75) * (x1 - x0))
        y_cut = float(y0 + rng.uniform(0.55, 0.75) * (y1 - y0))
        append_rectangle(boxes, x0=x0, x1=x1, y0=y0, y1=y_cut, height=height)
        append_rectangle(
            boxes,
            x0=x0,
            x1=x_cut,
            y0=y_cut,
            y1=y1,
            height=height * float(rng.uniform(0.85, 1.05)),
        )
    elif style == "u_shape":
        arm = float(rng.uniform(0.20, 0.32) * (x1 - x0))
        base = float(rng.uniform(0.22, 0.34) * (y1 - y0))
        append_rectangle(boxes, x0=x0, x1=x1, y0=y0, y1=y0 + base, height=height)
        append_rectangle(
            boxes, x0=x0, x1=x0 + arm, y0=y0 + base, y1=y1, height=height
        )
        append_rectangle(
            boxes, x0=x1 - arm, x1=x1, y0=y0 + base, y1=y1, height=height
        )
    else:
        gap = float(rng.uniform(2.0, 4.5))
        if rng.random() < 0.5:
            cut = float(rng.uniform(x0 + 0.35 * (x1 - x0), x0 + 0.65 * (x1 - x0)))
            append_rectangle(boxes, x0=x0, x1=cut - gap / 2, y0=y0, y1=y1, height=height)
            append_rectangle(
                boxes,
                x0=cut + gap / 2,
                x1=x1,
                y0=y0,
                y1=y1,
                height=height * float(rng.uniform(0.75, 1.15)),
            )
        else:
            cut = float(rng.uniform(y0 + 0.35 * (y1 - y0), y0 + 0.65 * (y1 - y0)))
            append_rectangle(boxes, x0=x0, x1=x1, y0=y0, y1=cut - gap / 2, height=height)
            append_rectangle(
                boxes,
                x0=x0,
                x1=x1,
                y0=cut + gap / 2,
                y1=y1,
                height=height * float(rng.uniform(0.75, 1.15)),
            )
    return boxes


def dense_scene_spec(
    rng: np.random.Generator,
    points: int,
) -> v374.SceneSpec:
    x_cells = int(rng.integers(5, 8))
    y_cells = int(rng.integers(5, 8))
    x_edges = random_grid_edges(rng, CITY_MIN, CITY_MAX, x_cells)
    y_edges = random_grid_edges(rng, CITY_MIN, CITY_MAX, y_cells)
    vertical_road = rng.uniform(6.0, 11.0, size=x_cells + 1)
    horizontal_road = rng.uniform(6.0, 11.0, size=y_cells + 1)
    boxes: list[np.ndarray] = []
    for ix in range(x_cells):
        for iy in range(y_cells):
            if rng.random() < 0.10:
                continue
            x0 = float(x_edges[ix] + 0.5 * vertical_road[ix])
            x1 = float(x_edges[ix + 1] - 0.5 * vertical_road[ix + 1])
            y0 = float(y_edges[iy] + 0.5 * horizontal_road[iy])
            y1 = float(y_edges[iy + 1] - 0.5 * horizontal_road[iy + 1])
            boxes.extend(block_buildings(rng, x0, x1, y0, y1))
    if len(boxes) < 30:
        raise RuntimeError("dense city generated too few building components")
    box_array = np.stack(boxes).astype(np.float32)
    # Mostly default-like concrete, with limited material diversity.
    material = rng.choice(
        np.arange(len(v374.MATERIALS)),
        size=len(box_array),
        p=np.asarray([0.70, 0.20, 0.08, 0.02]),
    ).astype(np.int8)

    positions = []
    attempts = 0
    while len(positions) < points and attempts < points * 1000:
        attempts += 1
        point = np.asarray(
            [rng.uniform(BS[0] + 2.0, 124.0), rng.uniform(-124.0, 122.0)],
            dtype=np.float32,
        )
        if v374.point_inside(point, box_array, margin=1.2):
            continue
        if not v374.point_inside(point, box_array, margin=2.5):
            continue
        positions.append(np.asarray([point[0], point[1], 1.5], dtype=np.float32))
    if len(positions) < points:
        raise RuntimeError("failed to sample enough one-sided free-space UE positions")
    return v374.SceneSpec(
        boxes=box_array,
        material=material,
        bs=BS.copy(),
        positions=np.stack(positions).astype(np.float32),
        frequency=FREQUENCY,
    )


def morphology_summary(specs: list[v374.SceneSpec]) -> dict[str, object]:
    import lifecycle_geometry_features as lgf

    feature = []
    building_count = []
    footprint_fraction = []
    for spec in specs:
        raster = lgf.raster_from_boxes(spec.boxes)
        feature.append(lgf.point_features(raster, spec.positions, spec.bs))
        building_count.append(len(spec.boxes))
        footprint_fraction.append(float(np.mean(raster.height > 0.2)))
    synthetic = np.concatenate(feature)
    output: dict[str, object] = {
        "scene_count": len(specs),
        "point_count": int(len(synthetic)),
        "building_component_mean": float(np.mean(building_count)),
        "footprint_fraction_mean": float(np.mean(footprint_fraction)),
        "point_feature_mean": np.mean(synthetic, axis=0).tolist(),
        "point_feature_std": np.std(synthetic, axis=0).tolist(),
        "local_density_mean": synthetic[:, [37, 39, 41]].mean(axis=0).tolist(),
        "local_density_std": synthetic[:, [37, 39, 41]].std(axis=0).tolist(),
    }
    if OFFICIAL_FEATURE_CACHE.is_file():
        official = np.load(OFFICIAL_FEATURE_CACHE)["feature"][:, :43]
        output["official_local_density_mean"] = official[:, [37, 39, 41]].mean(
            axis=0
        ).tolist()
        output["official_point_feature_standardized_rmse"] = float(
            np.sqrt(
                np.mean(
                    (
                        (np.mean(synthetic, axis=0) - np.mean(official, axis=0))
                        / np.maximum(np.std(official, axis=0), 0.05)
                    )
                    ** 2
                )
            )
        )
    return output


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCENE_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        dense_scene_spec(
            np.random.default_rng(args.seed + 10_000 * scene_id),
            args.points_per_scene,
        )
        for scene_id in range(args.scene_count)
    ]
    previous = None
    if REPORT.is_file() and not args.overwrite:
        previous = json.loads(REPORT.read_text(encoding="utf-8"))
    if (
        not args.morphology_audit_only
        and previous is not None
        and previous.get("status") == "morphology_audit_only"
        and previous.get("parameters", {}).get("seed") == args.seed
        and previous.get("parameters", {}).get("scene_count") == args.scene_count
        and previous.get("parameters", {}).get("points_per_scene")
        == args.points_per_scene
    ):
        morphology = previous["morphology"]
    else:
        morphology = morphology_summary(specs)
    if args.morphology_audit_only:
        report = {
            "method": VERSION,
            "status": "morphology_audit_only",
            "parameters": vars(args),
            "morphology": morphology,
            "outputs": [],
        }
        REPORT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(morphology, ensure_ascii=False, indent=2), flush=True)
        return

    import mitsuba as mi

    mi.set_variant(args.mitsuba_variant)
    from sionna.rt import (
        PathSolver,
        PlanarArray,
        Receiver,
        Transmitter,
        load_scene,
    )

    started = time.time()
    scene_reports = []
    output_paths = []
    for scene_id, spec in enumerate(specs):
        scene_dir = SCENE_ROOT / f"scene_{scene_id:03d}"
        output = scene_dir / "lifecycle.npz"
        scene_report_path = scene_dir / "report.json"
        if output.is_file() and scene_report_path.is_file() and not args.overwrite:
            scene_report = json.loads(scene_report_path.read_text(encoding="utf-8"))
        else:
            scene_xml = v374.write_scene(scene_dir, spec)
            payload, scene_report = v374.trace_scene(
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
            np.savez_compressed(output, **payload)
            scene_report["output"] = PATHS.relative(output)
            scene_report_path.write_text(
                json.dumps(scene_report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        output_paths.append(PATHS.relative(output))
        scene_reports.append(scene_report)
        print(
            f"v381 scene={scene_id} retained={scene_report['retained_family_count']} "
            f"active={scene_report['mean_active_families']:.2f}",
            flush=True,
        )

    report = {
        "method": VERSION,
        "status": "dense_procedural_sionna_dataset_generated",
        "scope": "independent dense-city lifecycle pretraining data",
        "compliance": {
            "uses": [
                "public official coordinate and point-cloud morphology statistics",
                "independent procedural geometry, UE coordinates, and Sionna PathSolver labels",
            ],
            "does_not_use": [
                "official train/test H as synthetic label",
                "restored Etoile mesh/channel, v212 teacher, hidden or online truth",
            ],
        },
        "parameters": vars(args),
        "morphology": morphology,
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
