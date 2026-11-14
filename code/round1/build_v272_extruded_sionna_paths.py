#!/usr/bin/env python3
"""Trace geometry-only Sionna paths on the v272 official-point-cloud mesh."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from project_paths import PATHS


VERSION_NUMBER = 272
VERSION = "v272_extruded_pointcloud_scene"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
SCENE_DIR = OUT_DIR / "scene"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MESH = SCENE_DIR / "extruded_buildings.obj"
GROUND = SCENE_DIR / "ground.obj"
SCENE_XML = SCENE_DIR / "scene.xml"
OUTPUT = OUT_DIR / "v272_extruded_sionna_paths.npz"
REPORT = REPORT_DIR / "v272_extruded_sionna_paths.json"
BS = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
LIGHT_SPEED = 299_792_458.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--samples-per-src", type=int, default=1_000_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=27201)
    parser.add_argument("--frequency", type=float, default=6.0e9)
    parser.add_argument("--subcarrier-spacing", type=float, default=30e3)
    parser.add_argument("--mitsuba-variant", default="llvm_ad_mono_polarized")
    return parser.parse_args()


def prepare_scene() -> None:
    if not MESH.is_file():
        raise FileNotFoundError(MESH)
    GROUND.write_text(
        "\n".join(
            [
                "# ground",
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
    SCENE_XML.write_text(
        "\n".join(
            [
                '<scene version="2.1.0">',
                '  <bsdf type="itu-radio-material" id="concrete">',
                '    <string name="type" value="concrete"/>',
                '    <float name="thickness" value="0.1"/>',
                "  </bsdf>",
                '  <shape type="obj" id="extruded-buildings">',
                '    <string name="filename" value="extruded_buildings.obj"/>',
                '    <boolean name="face_normals" value="true"/>',
                '    <ref id="concrete" name="bsdf"/>',
                "  </shape>",
                '  <shape type="obj" id="ground">',
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


def allocate(count: int, max_paths: int) -> dict[str, np.ndarray]:
    shape = (count, max_paths)
    return {
        "valid": np.zeros(shape, dtype=bool),
        "tau": np.full(shape, np.nan, dtype=np.float32),
        "path_length": np.full(shape, np.nan, dtype=np.float32),
        "delay_bin_physical": np.full(shape, np.nan, dtype=np.float32),
        "theta_t": np.full(shape, np.nan, dtype=np.float32),
        "phi_t": np.full(shape, np.nan, dtype=np.float32),
        "theta_r": np.full(shape, np.nan, dtype=np.float32),
        "phi_r": np.full(shape, np.nan, dtype=np.float32),
        "interaction_count": np.zeros(shape, dtype=np.int8),
        "interaction_mask": np.zeros(shape, dtype=np.int16),
        "score": np.full(shape, -np.inf, dtype=np.float32),
    }


def remove_receivers(scene, names: list[str]) -> None:
    for name in names:
        if scene.get(name) is not None:
            scene.remove(name)


def select_paths(paths, max_paths: int, max_depth: int, spacing: float) -> list[dict[str, np.ndarray]]:
    valid = paths.valid.numpy()[:, 0].astype(bool)
    tau = paths.tau.numpy()[:, 0].astype(np.float32)
    theta_t = paths.theta_t.numpy()[:, 0].astype(np.float32)
    phi_t = paths.phi_t.numpy()[:, 0].astype(np.float32)
    theta_r = paths.theta_r.numpy()[:, 0].astype(np.float32)
    phi_r = paths.phi_r.numpy()[:, 0].astype(np.float32)
    interactions = paths.interactions.numpy()[:, :, 0].transpose(1, 2, 0).astype(np.int8)
    output = []
    for row in range(len(valid)):
        ids = np.flatnonzero(valid[row] & np.isfinite(tau[row]) & (tau[row] >= 0.0))
        if len(ids) == 0:
            output.append({"ids": np.empty(0, dtype=np.int64)})
            continue
        interaction = interactions[row, ids, :max_depth]
        depth = np.sum(interaction != 0, axis=1).astype(np.int8)
        mask = np.bitwise_or.reduce(interaction.astype(np.int16), axis=1)
        length = tau[row, ids].astype(np.float64) * LIGHT_SPEED
        direct = depth == 0
        diffracted = (mask & 8) != 0
        rank = (
            4.0 * direct.astype(np.float64)
            + 0.35 * diffracted.astype(np.float64)
            - 0.35 * depth.astype(np.float64)
            - 0.006 * np.maximum(length - np.min(length), 0.0)
        )
        order = np.argsort(rank)[::-1][:max_paths]
        ids = ids[order]
        output.append(
            {
                "ids": ids,
                "tau": tau[row, ids],
                "path_length": length[order].astype(np.float32),
                "delay_bin_physical": (tau[row, ids] * spacing * 192.0).astype(np.float32),
                "theta_t": theta_t[row, ids],
                "phi_t": phi_t[row, ids],
                "theta_r": theta_r[row, ids],
                "phi_r": phi_r[row, ids],
                "interaction_count": depth[order],
                "interaction_mask": mask[order],
                "score": rank[order].astype(np.float32),
            }
        )
    return output


def trace_split(prefix, positions, args, scene, tx, mi, Receiver, PathSolver):
    arrays = allocate(len(positions), args.max_paths)
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
            scene.add(Receiver(name=name, position=mi.Point3f(*map(float, pos)), look_at=tx))
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
            diffuse_reflection=False,
            refraction=True,
            diffraction=True,
            edge_diffraction=True,
            diffraction_lit_region=True,
            seed=args.seed + (1_000_000 if prefix == "test" else 0) + start,
        )
        selected = select_paths(paths, args.max_paths, args.max_depth, args.subcarrier_spacing)
        counts = []
        for local, item in enumerate(selected):
            n = len(item["ids"])
            counts.append(n)
            if n == 0:
                continue
            row = start + local
            arrays["valid"][row, :n] = True
            for key in arrays:
                if key == "valid":
                    continue
                arrays[key][row, :n] = item[key]
        seconds = time.time() - t0
        batch_reports.append(
            {
                "start": start,
                "end": end,
                "seconds": seconds,
                "mean_paths": float(np.mean(counts)),
                "zero_paths": int(np.sum(np.asarray(counts) == 0)),
            }
        )
        print(
            f"v272 {prefix} {end}/{len(positions)} seconds={seconds:.3f} "
            f"paths={float(np.mean(counts)):.2f} zero={int(np.sum(np.asarray(counts) == 0))}",
            flush=True,
        )
    remove_receivers(scene, names)
    count = arrays["valid"].sum(axis=1)
    return arrays, {
        "count": len(positions),
        "seconds": time.time() - started,
        "path_count": {
            "mean": float(np.mean(count)),
            "median": float(np.median(count)),
            "p90": float(np.quantile(count, 0.9)),
            "zero": int(np.sum(count == 0)),
        },
        "batches": batch_reports,
    }


def main() -> None:
    args = parse_args()
    prepare_scene()
    import mitsuba as mi

    mi.set_variant(args.mitsuba_variant)
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)[: args.train_count]
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)[: args.test_count]
    scene = load_scene(str(SCENE_XML), merge_shapes=True)
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
        name="tx", position=mi.Point3f(*map(float, BS)), look_at=mi.Point3f(0.0, 0.0, 1.5)
    )
    scene.add(tx)
    train, train_report = trace_split("train", train_pos, args, scene, tx, mi, Receiver, PathSolver)
    test, test_report = trace_split("test", test_pos, args, scene, tx, mi, Receiver, PathSolver)
    np.savez_compressed(
        OUTPUT,
        train_pos=train_pos,
        test_pos=test_pos,
        bs_pos=BS,
        **{f"train_{key}": value for key, value in train.items()},
        **{f"test_{key}": value for key, value in test.items()},
    )
    report = {
        "method": VERSION,
        "status": "geometry_cache_only",
        "config": vars(args),
        "scene": PATHS.relative(SCENE_XML),
        "mesh": PATHS.relative(MESH),
        "train": train_report,
        "test": test_report,
        "outputs": {"paths": PATHS.relative(OUTPUT), "report": PATHS.relative(REPORT)},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
