#!/usr/bin/env python3
"""Freeze a full-data route-identity ensemble before strict local replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v279_joint_component_trajectory as v279
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v406_neural_propagation_mode_router as v406
import analyze_v411_multiview_contrastive_mode_retrieval as v411
import analyze_v412_path_sequence_identity_student as v412
import analyze_v413_crossmodal_path_identity_clip as v413
import analyze_v415_route_raster_identity_cnn as v415
import build_v415_route_raster_features as route_builder
from project_paths import PATHS


VERSION_NUMBER = 416
VERSION = "v416_full_route_ensemble_replay"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS = OUT_DIR / "v416_frozen_joint_targets.npz"
BASE = PATHS.cache_dir(284) / "v284_Round1_Test_Channel.npy"

FINAL_EPOCHS = 45
BATCH_SIZE = 32
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 3e-4
LOCAL_GATE = 0.645604
ACTIONS = (
    ("a025_t010", 0.25, 0.10),
    ("a025_t005", 0.25, 0.05),
    ("a050_t010", 0.50, 0.10),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("train", "project-score"), required=True
    )
    return parser.parse_args()


def seed_cache(rank: int) -> Path:
    return OUT_DIR / f"v416_seed{rank}.npz"


def normalize(value: np.ndarray) -> np.ndarray:
    return v406.normalize(np.asarray(value, dtype=np.float32))


def test_global_feature(
    fit_ids: np.ndarray,
    train_xy: np.ndarray,
    test_xy: np.ndarray,
    test_node: np.ndarray,
    target: np.ndarray,
    teacher: v411.SpectrumEncoder,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    distance, local = cKDTree(train_xy[fit_ids]).query(
        test_xy, k=v406.DONOR_K
    )
    donor_ids = fit_ids[np.asarray(local, dtype=np.int64)]
    anchor, _ = v406.weighted_anchor(target, donor_ids, distance)
    anchor_embedding = v411.encode_spectra(teacher, anchor, device)
    descriptor = v257.spectrum_descriptor(anchor, anchor.sum(axis=2))
    feature = np.concatenate(
        [test_node, anchor_embedding, descriptor], axis=1
    ).astype(np.float32)
    return anchor, donor_ids, feature


def train_final_model(
    fit_route: np.ndarray,
    fit_global: np.ndarray,
    teacher_embedding: np.ndarray,
    fit_ids: np.ndarray,
    fit_donor: np.ndarray,
    donor_utility: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[v415.RouteIdentityCNN, list[dict[str, float]]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    route_tensor = torch.from_numpy(fit_route).to(device)
    global_tensor = torch.from_numpy(fit_global).to(device)
    teacher_tensor = torch.from_numpy(teacher_embedding).to(device)
    fit_id_tensor = torch.from_numpy(fit_ids).to(device)
    donor_id_tensor = torch.from_numpy(fit_donor).to(device)
    utility_tensor = torch.from_numpy(donor_utility).to(device)
    model = v415.RouteIdentityCNN(fit_global.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history: list[dict[str, float]] = []
    rows = np.arange(len(fit_ids), dtype=np.int64)
    for epoch in range(1, FINAL_EPOCHS + 1):
        model.train()
        order = rng.permutation(rows)
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            rows_np = order[start : start + BATCH_SIZE]
            batch = torch.from_numpy(rows_np).to(device)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                geometry_embedding = model.encode_geometry(
                    route_tensor[batch], global_tensor[batch]
                )
                query_spectrum = model.encode_spectrum(
                    teacher_tensor[fit_id_tensor[batch]]
                )
                donor_spectrum = model.encode_spectrum(
                    teacher_tensor[donor_id_tensor[batch]]
                )
                loss = v413.crossmodal_loss(
                    geometry_embedding,
                    query_spectrum,
                    donor_spectrum,
                    utility_tensor[batch],
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(rows_np)
        if epoch == 1 or epoch % 5 == 0 or epoch == FINAL_EPOCHS:
            history.append(
                {"epoch": epoch, "loss": total_loss / float(len(fit_ids))}
            )
            print(
                f"v416 seed={seed} epoch={epoch} loss={history[-1]['loss']:.6f}",
                flush=True,
            )
    return model, history


def train_seed(rank: int, device: torch.device) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    targets = np.load(v412.TARGET_CACHE)
    target = normalize(targets["sh"])
    views = v411.make_views(targets)
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(
        targets["valid"].astype(bool) & component["valid"].any(axis=1)
    ).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    train_node, test_node = v257.build_node_features(train_pos[:, :2], test_pos[:, :2])

    teacher_seed = 416_000 + rank
    student_seed = 416_100 + rank
    teacher, teacher_history = v411.train_teacher(
        valid_ids, views, teacher_seed, device
    )
    teacher_embedding = np.zeros((len(target), v411.EMBED_DIM), dtype=np.float32)
    teacher_embedding[valid_ids] = v411.encode_spectra(
        teacher, target[valid_ids], device
    )
    fit_anchor, fit_donor, fit_global = v412.make_global_feature(
        valid_ids,
        valid_ids,
        train_pos[:, :2],
        train_node,
        target,
        teacher,
        device,
    )
    test_anchor, test_donor, test_global = test_global_feature(
        valid_ids,
        train_pos[:, :2],
        test_pos[:, :2],
        test_node,
        target,
        teacher,
        device,
    )
    fit_global, test_global = v412.standardize_global(
        fit_global,
        test_global,
        np.arange(len(valid_ids), dtype=np.int64),
    )
    utility, utility_diagnostic = v413.stable_donor_utility(
        valid_ids,
        valid_ids,
        fit_donor,
        train_pos[:, :2],
        views,
        target,
    )
    route_cache = np.load(route_builder.CACHE)
    train_route = route_cache["train"]
    test_route = route_cache["test"]
    model, student_history = train_final_model(
        train_route[valid_ids],
        fit_global,
        teacher_embedding,
        valid_ids,
        fit_donor,
        utility,
        student_seed,
        device,
    )
    projected_teacher = v415.project_spectrum(model, teacher_embedding, device)
    test_geometry = v415.predict_geometry(
        model,
        torch.from_numpy(test_route).to(device),
        torch.from_numpy(test_global).to(device),
    )
    rank_index, margin = v411.retrieve(
        test_geometry, test_donor, projected_teacher
    )
    frozen = {
        name: v411.render(
            test_anchor,
            test_donor,
            rank_index,
            margin,
            target,
            alpha,
            threshold,
        )
        for name, alpha, threshold in ACTIONS
    }
    np.savez_compressed(
        seed_cache(rank),
        **frozen,
        margin=margin,
        selected_donor=test_donor[np.arange(len(test_donor)), rank_index],
        valid_ids=valid_ids,
        teacher_seed=np.asarray(teacher_seed),
        student_seed=np.asarray(student_seed),
        final_epochs=np.asarray(FINAL_EPOCHS),
        teacher_history=np.asarray(json.dumps(teacher_history)),
        student_history=np.asarray(json.dumps(student_history)),
        utility_diagnostic=np.asarray(json.dumps(utility_diagnostic)),
    )
    print(f"v416 seed rank={rank} frozen at {seed_cache(rank)}", flush=True)


def aggregate_seed_targets(world_size: int) -> None:
    parts = [np.load(seed_cache(rank)) for rank in range(world_size)]
    payload: dict[str, np.ndarray] = {}
    for name, _, _ in ACTIONS:
        stacked = np.stack([part[name] for part in parts], axis=0)
        payload[name] = normalize(stacked.mean(axis=0, dtype=np.float64))
    payload["seed_count"] = np.asarray(world_size, dtype=np.int64)
    payload["actions_json"] = np.asarray(json.dumps(ACTIONS))
    np.savez_compressed(TARGETS, **payload)
    report = {
        "method": VERSION,
        "status": "frozen_targets_pending_local_replay",
        "primary_hypothesis": (
            "A multi-seed dense official-pointcloud route encoder may recover enough "
            "stable train-H propagation identity to improve the strict v284 carrier."
        ),
        "compliance": {
            "training_uses": [
                "official train H",
                "official train/test coordinates",
                "official Round1_Map.ply derived route rasters",
            ],
            "training_does_not_use": [
                "synthetic test truth",
                "v212 output or teacher",
                "online labels",
                "raw Sionna channel coefficients",
            ],
            "local_truth_read_only_after_all_targets_are_frozen": True,
        },
        "training": {
            "seed_count": world_size,
            "final_epochs": FINAL_EPOCHS,
            "actions": [
                {"name": name, "alpha": alpha, "threshold": threshold}
                for name, alpha, threshold in ACTIONS
            ],
        },
        "outputs": {
            "targets": PATHS.relative(TARGETS),
            "predictions": {},
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"v416 all frozen targets written to {TARGETS}", flush=True)


def distributed_train() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("v416 final ensemble requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    train_seed(local_rank, torch.device("cuda", local_rank))
    if world_size > 1:
        dist.barrier()
    if local_rank == 0:
        aggregate_seed_targets(world_size)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def project_and_score() -> None:
    if not TARGETS.is_file():
        raise FileNotFoundError(TARGETS)
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    target_cache = np.load(TARGETS)
    outputs: dict[str, Path] = {}
    projection_summary: dict[str, dict[str, float]] = {}
    v247.BASE = BASE
    # Freeze every full-H output before reading local truth.
    for name, _, _ in ACTIONS:
        target_sh = target_cache[name].astype(np.float32)
        target_d = v279.derive_d_numpy(target_sh)
        output = OUT_DIR / f"v416_{name}_Round1_Test_Channel.npy"
        diagnostic = v247.project(target_sh, target_d, output)
        outputs[name] = output
        projection_summary[name] = {
            key: float(np.mean(value)) for key, value in diagnostic.items()
        }
    scores = {name: v208.score_prediction(path) for name, path in outputs.items()}
    baseline = v208.score_prediction(BASE)
    best_name = max(scores, key=lambda name: scores[name]["score_traceback_leaked"])
    best_score = float(scores[best_name]["score_traceback_leaked"])
    passed = bool(best_score > LOCAL_GATE)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report.update(
        {
            "status": (
                "local_gate_passed_ready_for_packaging"
                if passed
                else "analysis_only_local_gate_failed"
            ),
            "projection": projection_summary,
            "strict_local_replay": {
                "gate": LOCAL_GATE,
                "baseline_v284": baseline,
                "actions": scores,
                "best_action": best_name,
                "best_score": best_score,
                "passed": passed,
            },
        }
    )
    report["outputs"]["predictions"] = {
        name: PATHS.relative(path) for name, path in outputs.items()
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(report["strict_local_replay"], ensure_ascii=False, indent=2),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    for path in (
        v412.TARGET_CACHE,
        v297.COMPONENT_CACHE,
        route_builder.CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.stage == "train":
        distributed_train()
    else:
        project_and_score()


if __name__ == "__main__":
    main()
