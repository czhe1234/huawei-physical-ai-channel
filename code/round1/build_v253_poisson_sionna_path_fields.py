#!/usr/bin/env python3
"""v253: trace scalar path fields on the official Poisson mesh."""

from __future__ import annotations

import argparse
import json

import numpy as np

import build_v251_pointcloud_sionna_paths as v251
import build_v252_pointcloud_sionna_path_fields as v252
from project_paths import PATHS


VERSION_NUMBER = 253
VERSION = "v253_poisson_sionna_path_fields"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
SCENE_DIR = OUT_DIR / "scene"
MESH = SCENE_DIR / "official_poisson_mesh.ply"
SCENE_XML = SCENE_DIR / "poisson_scene.xml"
OUTPUT = OUT_DIR / f"{VERSION}.npz"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}_build.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--samples-per-src", type=int, default=300_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=25301)
    parser.add_argument("--frequency", type=float, default=6.0e9)
    parser.add_argument("--subcarrier-spacing", type=float, default=30e3)
    parser.add_argument("--mitsuba-variant", default="llvm_ad_mono_polarized")
    parser.add_argument(
        "--edge-diffraction",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def prepare_xml() -> None:
    if not MESH.is_file():
        raise FileNotFoundError(MESH)
    SCENE_XML.write_text(
        "\n".join(
            [
                '<scene version="2.1.0">',
                '  <bsdf type="itu-radio-material" id="concrete">',
                '    <string name="type" value="concrete"/>',
                '    <float name="thickness" value="0.1"/>',
                "  </bsdf>",
                '  <shape type="ply" id="mesh-official-poisson">',
                '    <string name="filename" value="official_poisson_mesh.ply"/>',
                '    <boolean name="face_normals" value="true"/>',
                '    <ref id="concrete" name="bsdf"/>',
                "  </shape>",
                "</scene>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_xml()

    import mitsuba as mi

    mi.set_variant(args.mitsuba_variant)
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    train_all = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_all = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    train_pos = train_all[: min(args.train_count, len(train_all))]
    test_pos = test_all[: min(args.test_count, len(test_all))]
    print(f"v253 loading Poisson scene {SCENE_XML}", flush=True)
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
        name="tx",
        position=mi.Point3f(*map(float, v251.BS_POS)),
        look_at=mi.Point3f(0.0, 0.0, 1.5),
    )
    scene.add(tx)
    train, train_report = v252.trace_split(
        "train", train_pos, args, scene, tx, mi, Receiver, PathSolver
    )
    test, test_report = v252.trace_split(
        "test", test_pos, args, scene, tx, mi, Receiver, PathSolver
    )
    payload: dict[str, np.ndarray] = {
        "train_pos": train_pos,
        "test_pos": test_pos,
        "bs_pos": v251.BS_POS,
    }
    for prefix, arrays in (("train", train), ("test", test)):
        for key, value in arrays.items():
            payload[f"{prefix}_{key}"] = value
    np.savez_compressed(OUTPUT, **payload)
    report = {
        "method": VERSION,
        "status": "physics_feature_cache_only",
        "primary_hypothesis": (
            "The official Poisson mesh provides more accurate visibility and path fields "
            "than disconnected point-cloud wall rectangles."
        ),
        "compliance_boundary": {
            "uses": [
                "official Round1_Map.ply Poisson mesh",
                "official train/test coordinates",
                "Sionna scalar path fields",
            ],
            "does_not_use": [
                "Paths.cfr or MIMO RT H",
                "restored Etoile mesh/object/material labels",
                "synthetic channel labels",
                "v212/v217 teacher outputs",
                "local replay truth",
                "hidden online labels",
            ],
        },
        "config": vars(args),
        "train": train_report,
        "test": test_report,
        "outputs": {
            "features": PATHS.relative(OUTPUT),
            "scene": PATHS.relative(SCENE_XML),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
