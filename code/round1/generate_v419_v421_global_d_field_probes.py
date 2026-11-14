#!/usr/bin/env python3
"""Train and package three carrier-relative global-D field probes."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from scipy.spatial import cKDTree

import analyze_v247_delaunay_spectral_field as v247
import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v412_path_sequence_identity_student as v412
import analyze_v417_multiresolution_global_d_field as v417
import generate_v396_confocal_mass_preserving_d_projection as v396
from project_paths import PATHS


SUITE = (
    (419, "v419_global_d_field_relative_a005", 0.05),
    (420, "v420_global_d_field_relative_a010", 0.10),
    (421, "v421_global_d_field_relative_a020", 0.20),
)
BASE = PATHS.cache_dir(284) / "v284_Round1_Test_Channel.npy"
SHARED_OUT = PATHS.cache_dir(419)
CHECKPOINT_DIR = PATHS.checkpoints / "v419_v421_global_d_field"
FROZEN = SHARED_OUT / "v419_v421_frozen_global_d_field.npz"
SEED_COUNT = 5

SOURCE_FILES = [
    "generate_v419_v421_global_d_field_probes.py",
    "analyze_v417_multiresolution_global_d_field.py",
    "analyze_v414_multiresolution_identity_field.py",
    "analyze_v412_path_sequence_identity_student.py",
    "analyze_v411_multiview_contrastive_mode_retrieval.py",
    "analyze_v406_neural_propagation_mode_router.py",
    "analyze_v402_visible_path_residual_factorization.py",
    "analyze_v315_full_sequence_confirmatory.py",
    "analyze_v297_continuous_delay_component_transport.py",
    "analyze_v257_neural_pairwise_spectral_transport.py",
    "analyze_v247_delaunay_spectral_field.py",
    "generate_v396_confocal_mass_preserving_d_projection.py",
    "generate_v187_leaked_pas_lifecycle_projection.py",
    "generate_v87_easy_field_projection.py",
    "powerfield_common.py",
    "project_paths.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "package"), required=True)
    return parser.parse_args()


def seed_file(rank: int) -> Path:
    return SHARED_OUT / f"v419_v421_seed{rank}.npz"


def checkpoint_file(rank: int) -> Path:
    return CHECKPOINT_DIR / f"seed_{41900 + rank}.pt"


def test_anchor(
    train_xy: np.ndarray,
    test_xy: np.ndarray,
    valid_ids: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    distance, local = cKDTree(train_xy[valid_ids]).query(
        test_xy, k=v417.NEIGHBORS
    )
    donor = valid_ids[local]
    weight = 1.0 / np.maximum(distance, v417.DISTANCE_FLOOR) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return v417.normalize(
        np.einsum("qk,qkd->qd", weight, target[donor], optimize=True)
    )


def train_seed(rank: int, device: torch.device) -> None:
    SHARED_OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    target_cache = np.load(v412.TARGET_CACHE)
    target = v417.normalize(target_cache["d"])
    valid_ids = np.flatnonzero(target_cache["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    train_xy = train_pos[:, :2]
    test_xy = test_pos[:, :2]
    train_node, test_node = v257.build_node_features(train_xy, test_xy)
    fit_anchor = v417.knn_anchor(valid_ids, valid_ids, train_xy, target)
    query_anchor = test_anchor(train_xy, test_xy, valid_ids, target)
    mean = train_node[valid_ids].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_node[valid_ids].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    fit_node = ((train_node[valid_ids] - mean) / std).astype(np.float32)
    query_node = ((test_node - mean) / std).astype(np.float32)
    all_xy = np.concatenate([train_xy, test_xy], axis=0)
    bounds = (
        all_xy.min(axis=0).astype(np.float32) - 1.0,
        all_xy.max(axis=0).astype(np.float32) + 1.0,
    )
    seed = 41900 + rank
    model, history = v417.train_model(
        train_xy[valid_ids],
        fit_node,
        fit_anchor,
        target[valid_ids],
        bounds,
        seed,
        device,
    )
    proposal = v417.predict(model, test_xy, query_node, query_anchor, device)
    torch.save(
        {
            "version": "v419_v421_global_d_field",
            "seed": seed,
            "state_dict": model.state_dict(),
            "node_mean": mean,
            "node_std": std,
            "lower": bounds[0],
            "upper": bounds[1],
            "epochs": v417.EPOCHS,
            "resolutions": v417.RESOLUTIONS,
            "grid_dim": v417.GRID_DIM,
        },
        checkpoint_file(rank),
    )
    np.savez_compressed(
        seed_file(rank),
        proposal=proposal,
        anchor=query_anchor,
        history=np.asarray(json.dumps(history)),
        seed=np.asarray(seed),
    )
    print(f"v419-v421 rank={rank} frozen", flush=True)


def aggregate() -> None:
    parts = [np.load(seed_file(rank)) for rank in range(SEED_COUNT)]
    anchor = parts[0]["anchor"].astype(np.float32)
    for part in parts[1:]:
        if not np.allclose(anchor, part["anchor"], atol=1e-7):
            raise AssertionError("seed anchors differ")
    proposal = v417.normalize(
        np.stack([part["proposal"] for part in parts], axis=0).mean(
            axis=0, dtype=np.float64
        )
    )
    log_ratio = np.clip(
        np.log(np.maximum(proposal, 1e-10))
        - np.log(np.maximum(anchor, 1e-10)),
        -4.0,
        4.0,
    ).astype(np.float32)
    np.savez_compressed(
        FROZEN,
        anchor=anchor,
        proposal=proposal,
        log_ratio=log_ratio,
        seed_count=np.asarray(SEED_COUNT),
        oof_relative_alpha=np.asarray([0.05, 0.10, 0.20], dtype=np.float32),
        oof_relative_delta=np.asarray(
            [0.001694, 0.002988, 0.004449], dtype=np.float32
        ),
    )
    print(f"v419-v421 frozen ensemble written to {FROZEN}", flush=True)


def distributed_train() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != SEED_COUNT:
        raise ValueError(f"train stage requires exactly {SEED_COUNT} workers")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    train_seed(local_rank, torch.device("cuda", local_rank))
    dist.barrier()
    if local_rank == 0:
        aggregate()
    dist.barrier()
    dist.destroy_process_group()


def current_d() -> np.ndarray:
    source = np.load(BASE, mmap_mode="r")
    output = np.empty((len(source), 192), dtype=np.float32)
    for start in range(0, len(source), 8):
        end = min(start + 8, len(source))
        _sh, output[start:end] = v247.extract_current(
            np.asarray(source[start:end], dtype=np.complex64)
        )
    return v417.normalize(output)


def package_one(
    version_number: int,
    version: str,
    alpha: float,
    current: np.ndarray,
    log_ratio: np.ndarray,
) -> None:
    out_dir = PATHS.cache_dir(version_number)
    report_dir = PATHS.version_report_dir(version_number)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"v{version_number}_Round1_Test_Channel.npy"
    targets = out_dir / f"v{version_number}_global_d_field_targets.npz"
    report_path = report_dir / f"{version}.json"
    zip_path = PATHS.round1_versions / f"{version}_candidate_pending.zip"
    target = v417.normalize(current * np.exp(alpha * log_ratio))
    np.savez_compressed(
        targets,
        current_d=current,
        target_d=target,
        log_ratio=log_ratio,
        effective_alpha=np.asarray(alpha, dtype=np.float32),
    )
    v396.BASE = BASE
    v396.OUTPUT = output
    projection = v396.project_d_only(target)
    output_check = v396.validate_output()

    model_files = [checkpoint_file(rank) for rank in range(SEED_COUNT)] + [
        FROZEN,
        targets,
        PATHS.version_report_dir(417) / "v417_multiresolution_global_d_field.json",
        PATHS.cache_dir(417) / "v417_multiresolution_global_d_field_oof.npz",
        PATHS.checkpoints / "v284_planar_geometry_residual_field" / "seed_28401.pt",
        PATHS.checkpoints / "v284_planar_geometry_residual_field" / "seed_28431.pt",
        PATHS.checkpoints / "v284_planar_geometry_residual_field" / "seed_28461.pt",
        PATHS.cache_dir(284) / "v284_planar_geometry_targets.npz",
    ]
    for filename in SOURCE_FILES:
        if not (PATHS.code / filename).is_file():
            raise FileNotFoundError(PATHS.code / filename)
    for path in model_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(output, "Round1_Test_Channel.npy")
        for filename in SOURCE_FILES:
            archive.write(PATHS.code / filename, filename)
        for path in model_files:
            archive.write(path, path.name)
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    nested = [name for name in members if "/" in name.strip("/")]
    if nested:
        raise ValueError(f"submission ZIP must be flat: {nested}")
    report = {
        "method": version,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The v417 global-D field has a 5/5-fold positive carrier-relative direction; "
            "a bounded relative action can improve the strict v284 carrier online."
        ),
        "official_oof_evidence": {
            "alpha": alpha,
            "relative_cosine_delta": {
                0.05: 0.001694,
                0.10: 0.002988,
                0.20: 0.004449,
            }[alpha],
            "positive_folds": 5,
        },
        "parameters": {
            "seed_count": SEED_COUNT,
            "effective_alpha": alpha,
            "action": "carrier-relative log-D residual",
            "base": PATHS.relative(BASE),
        },
        "compliance_boundary": {
            "uses": [
                "official train H normalized D",
                "official train/test coordinates",
                "official pointcloud-derived node features",
                "strict official-only v284 carrier",
            ],
            "does_not_use": [
                "v207/v212 synthetic test target or teacher",
                "online labels",
                "restored scene/material/object labels",
                "raw RT channel output",
            ],
        },
        "target_diagnostics": {
            "target_current_l1": v396.summarize(np.sum(np.abs(target - current), axis=1)),
            "target_current_cosine": v396.summarize(v417.row_cos(target, current)),
            "finite": bool(np.isfinite(target).all()),
            "row_sum_max_abs_error": float(np.max(np.abs(target.sum(axis=1) - 1.0))),
        },
        "projection_diagnostics": projection,
        "integrity": {
            **output_check,
            "zip_crc": True,
            "zip_member_count": len(members),
            "sha256": v396.digest(zip_path, "sha256"),
            "md5": v396.digest(zip_path, "md5"),
        },
        "outputs": {
            "prediction": PATHS.relative(output),
            "targets": PATHS.relative(targets),
            "zip": PATHS.relative(zip_path),
            "report": PATHS.relative(report_path),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def package_suite() -> None:
    if not FROZEN.is_file():
        raise FileNotFoundError(FROZEN)
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    frozen = np.load(FROZEN)
    log_ratio = frozen["log_ratio"].astype(np.float32)
    current = current_d()
    for version_number, version, alpha in SUITE:
        package_one(version_number, version, alpha, current, log_ratio)


def main() -> None:
    args = parse_args()
    if args.stage == "train":
        distributed_train()
    else:
        package_suite()


if __name__ == "__main__":
    main()
