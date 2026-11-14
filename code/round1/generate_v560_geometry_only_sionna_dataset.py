#!/usr/bin/env python3
"""Generate v560 Sionna paths with power-independent candidate selection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import generate_v437_single_gpu_sionna_pretraining_dataset as generator
from project_paths import PATHS


VERSION_NUMBER = 560
PILOT_POSITION_SOURCE = PATHS.cache_dir(442) / "positions.npz"
OFFICIAL_TRAIN = PATHS.round1 / "Round1_Train_Pos.npy"
OFFICIAL_TEST = PATHS.round1 / "Round1_Test_Pos.npy"
BASE_EMPTY_PATH_TARGETS = generator.empty_path_targets


def parse_args() -> argparse.Namespace:
    dataset = os.environ.get("HW_V560_DATASET", "pilot")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pilot", "official"), default=dataset)
    parser.add_argument("--count", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--max-paths", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--samples-per-src", type=int, default=1_000_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=1_000_000)
    parser.add_argument("--frequency", type=float, default=6.693e9)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--min-test-distance", type=float, default=1.0)
    parser.add_argument("--min-train-distance", type=float, default=0.05)
    parser.add_argument(
        "--save-full-h", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-shards", action="store_true")
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    args = parser.parse_args()
    if args.dataset == "pilot":
        args.count = 20_000 if args.count is None else args.count
        args.shard_size = 1_000 if args.shard_size is None else args.shard_size
        args.seed = 44_201 if args.seed is None else args.seed
    else:
        args.count = 2_500 if args.count is None else args.count
        args.shard_size = 100 if args.shard_size is None else args.shard_size
        args.seed = 475_001 if args.seed is None else args.seed
        if args.count != 2_500:
            raise ValueError("The official v560 dataset must contain all 2500 rows")
        args.min_test_distance = 0.0
        args.min_train_distance = 0.0
    return args


def dataset_paths(dataset: str) -> tuple[Path, Path, Path]:
    root = PATHS.cache_dir(VERSION_NUMBER) / dataset
    report = (
        PATHS.version_report_dir(VERSION_NUMBER)
        / f"v560_{dataset}_geometry_only_sionna_dataset.json"
    )
    return root, root / "positions.npz", report


def prepare_position_manifest(
    dataset: str,
    count: int,
    path: Path,
    *,
    overwrite: bool,
) -> dict[str, np.ndarray]:
    if path.is_file() and not overwrite:
        cached = np.load(path)
        if len(cached["positions"]) != count:
            raise ValueError(
                f"Existing {path} has {len(cached['positions'])} rows, requested {count}"
            )
        return {name: cached[name] for name in cached.files}

    if dataset == "pilot":
        if not PILOT_POSITION_SOURCE.is_file():
            raise FileNotFoundError(PILOT_POSITION_SOURCE)
        source = np.load(PILOT_POSITION_SOURCE)
        if count > len(source["positions"]):
            raise ValueError(f"Requested {count} pilot rows from {len(source['positions'])}")
        payload = {name: source[name][:count] for name in source.files}
        payload["source_position_id"] = np.arange(count, dtype=np.int64)
    else:
        train = np.load(OFFICIAL_TRAIN).astype(np.float32)
        test = np.load(OFFICIAL_TEST).astype(np.float32)
        positions = np.concatenate([train, test], axis=0)
        official_split = np.concatenate(
            [np.zeros(len(train), dtype=np.int8), np.ones(len(test), dtype=np.int8)]
        )
        payload = {
            "positions": positions,
            "sample_mode": official_split,
            "pretrain_split": official_split,
            "min_official_test_distance": np.zeros(len(positions), dtype=np.float32),
            "min_official_train_distance": np.zeros(len(positions), dtype=np.float32),
            "official_split": official_split,
            "source_position_id": np.arange(len(positions), dtype=np.int64),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".{os.getpid()}.partial.npz")
    np.savez(temporary, **payload)
    temporary.replace(path)
    return payload


def empty_path_targets(
    count: int,
    max_paths: int,
    max_depth: int,
) -> dict[str, np.ndarray]:
    output = BASE_EMPTY_PATH_TARGETS(count, max_paths, max_depth)
    output.update(
        {
            "raw_valid_count": np.zeros(count, dtype=np.int32),
            "eligible_path_count": np.zeros(count, dtype=np.int32),
            "selected_path_count": np.zeros(count, dtype=np.int32),
            "retained_power_fraction": np.ones(count, dtype=np.float32),
            "power_topk_overlap": np.ones(count, dtype=np.float32),
        }
    )
    return output


def geometry_order(
    ids: np.ndarray,
    tau: np.ndarray,
    interaction_count: np.ndarray,
    phi_t: np.ndarray,
    phi_r: np.ndarray,
) -> np.ndarray:
    """Match v445's token order without consulting any path coefficient."""
    local = np.lexsort(
        (
            ids,
            phi_r[ids].astype(np.float64),
            phi_t[ids].astype(np.float64),
            interaction_count[ids].astype(np.int16),
            tau[ids].astype(np.float64),
        )
    )
    return ids[local]


def extract_path_targets(paths, max_paths: int, max_depth: int) -> dict[str, np.ndarray]:
    valid = paths.valid.numpy()[:, 0].astype(bool)
    tau = paths.tau.numpy()[:, 0].astype(np.float32)
    theta_t = paths.theta_t.numpy()[:, 0].astype(np.float32)
    phi_t = paths.phi_t.numpy()[:, 0].astype(np.float32)
    theta_r = paths.theta_r.numpy()[:, 0].astype(np.float32)
    phi_r = paths.phi_r.numpy()[:, 0].astype(np.float32)
    interactions = paths.interactions.numpy()[:, :, 0].transpose(1, 2, 0).astype(np.int8)
    objects = paths.objects.numpy()[:, :, 0].transpose(1, 2, 0).astype(np.int64)
    objects = np.where(objects >= np.iinfo(np.uint32).max, -1, objects).astype(np.int32)
    vertices = paths.vertices.numpy()[:, :, 0].transpose(1, 2, 0, 3).astype(np.float32)
    interaction_count = np.sum(interactions != 0, axis=2).astype(np.int16)

    # Coefficients are labels only. They never affect eligibility, ordering, or membership.
    cir, _ = paths.cir(normalize_delays=False, out_type="numpy")
    raw = cir[:, :, 0, :, :, 0].astype(np.complex64)
    element_power = (np.abs(raw) ** 2).astype(np.float32)
    path_power = np.sum(element_power, axis=(1, 2), dtype=np.float64)
    output = empty_path_targets(len(valid), max_paths, max_depth)

    for row in range(len(valid)):
        geometry_finite = (
            np.isfinite(tau[row])
            & np.isfinite(theta_t[row])
            & np.isfinite(phi_t[row])
            & np.isfinite(theta_r[row])
            & np.isfinite(phi_r[row])
        )
        ids = np.flatnonzero(valid[row] & geometry_finite & (tau[row] >= 0.0))
        output["raw_valid_count"][row] = int(np.sum(valid[row]))
        output["eligible_path_count"][row] = len(ids)
        if not len(ids):
            continue

        ordered = geometry_order(
            ids,
            tau[row],
            interaction_count[row],
            phi_t[row],
            phi_r[row],
        )
        selected = ordered[:max_paths]
        count = len(selected)
        output["selected_path_count"][row] = count
        output["valid"][row, :count] = True
        output["tau"][row, :count] = tau[row, selected]
        output["theta_t"][row, :count] = theta_t[row, selected]
        output["phi_t"][row, :count] = phi_t[row, selected]
        output["theta_r"][row, :count] = theta_r[row, selected]
        output["phi_r"][row, :count] = phi_r[row, selected]

        all_power = np.where(
            np.isfinite(path_power[row, ids]) & (path_power[row, ids] > 0.0),
            path_power[row, ids],
            0.0,
        )
        selected_power = np.where(
            np.isfinite(path_power[row, selected]) & (path_power[row, selected] > 0.0),
            path_power[row, selected],
            0.0,
        )
        all_mass = float(np.sum(all_power, dtype=np.float64))
        selected_mass = float(np.sum(selected_power, dtype=np.float64))
        output["retained_power_fraction"][row] = (
            selected_mass / all_mass if all_mass > 0.0 else 1.0
        )
        power_order = ids[np.argsort(path_power[row, ids], kind="stable")[::-1]][:max_paths]
        output["power_topk_overlap"][row] = (
            len(np.intersect1d(selected, power_order, assume_unique=False))
            / max(len(selected), 1)
        )
        output["log_power"][row, :count] = np.log(
            np.maximum(selected_power, 1e-30)
        ).astype(np.float32)
        output["normalized_power"][row, :count] = (
            selected_power / max(selected_mass, 1e-30)
        ).astype(np.float16)

        selected_interactions = interactions[row, selected, :max_depth]
        output["interactions"][row, :count] = selected_interactions
        output["objects"][row, :count] = objects[row, selected, :max_depth]
        output["vertices"][row, :count] = vertices[row, selected, :max_depth]
        output["interaction_count"][row, :count] = interaction_count[row, selected]
        output["interaction_mask"][row, :count] = np.bitwise_or.reduce(
            selected_interactions.astype(np.int16), axis=1
        )
        output["complex_reference"][row, :count] = raw[row, 0, 0, selected]

        selected_element = element_power[row][:, :, selected].reshape(
            4, 2, 16, 8, count
        )
        denominator = np.maximum(selected_power, 1e-30)
        output["rx_fingerprint"][row, :count] = (
            np.sum(selected_element, axis=(1, 2, 3), dtype=np.float64).T
            / denominator[:, None]
        ).astype(np.float16)
        output["tx_pol_fingerprint"][row, :count] = (
            np.sum(selected_element, axis=(0, 2, 3), dtype=np.float64).T
            / denominator[:, None]
        ).astype(np.float16)
        output["tx_h_fingerprint"][row, :count] = (
            np.sum(selected_element, axis=(0, 1, 3), dtype=np.float64).T
            / denominator[:, None]
        ).astype(np.float16)
        output["tx_v_fingerprint"][row, :count] = (
            np.sum(selected_element, axis=(0, 1, 2), dtype=np.float64).T
            / denominator[:, None]
        ).astype(np.float16)
    return output


def configure(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root, positions, report = dataset_paths(args.dataset)
    generator.VERSION_NUMBER = VERSION_NUMBER
    generator.VERSION = f"v560_{args.dataset}_geometry_only_sionna_dataset"
    generator.OUT_DIR = root
    generator.SHARD_ROOT = root / "shards"
    generator.POSITIONS_PATH = positions
    generator.REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
    generator.REPORT = report
    generator.empty_path_targets = empty_path_targets
    generator.extract_path_targets = extract_path_targets
    generator.generate_positions = lambda inner: prepare_position_manifest(
        args.dataset,
        inner.count,
        positions,
        overwrite=inner.overwrite,
    )
    generator.parse_args = lambda: args
    return root, positions, report


def main() -> None:
    args = parse_args()
    _, _, report_path = configure(args)
    generator.main()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["candidate_contract"] = {
        "membership_inputs": [
            "paths.valid",
            "finite nonnegative delay",
            "finite AoD/AoA",
        ],
        "ordering": "tau, interaction_count, phi_t, phi_r, raw path id",
        "forbidden_membership_inputs": [
            "path power",
            "complex coefficient",
            "power rank",
            "port fingerprint",
        ],
        "coefficient_use": "labels and retained-energy audit only",
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
