#!/usr/bin/env python3
"""Train and score the strict E200 finalist-path calibrator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any

import numpy as np
import torch

import train_e190_official_sionna_path_calibrator as e190
import train_e195_directional_probe_calibrator as e195
import train_surface_interaction_energy_model as e85
from e200_config_search import portable_payload
from prepare_e200_candidate_cache import (
    CACHE_DIR,
    COORDINATE,
    FEATURES,
    PORT_PRIOR,
    REPORT as CACHE_REPORT,
    SEQUENCE_GEOMETRY,
    SEQUENCE_HASH,
)
from project_paths import PATHS


NAMESPACE = os.environ.get("HW_E200_NAMESPACE", "e200").strip().lower()
if NAMESPACE not in {"e200", "e201"}:
    raise ValueError(f"unsupported E200-compatible namespace: {NAMESPACE}")
VERSION = f"Round2-{NAMESPACE.upper()}-finalist-path-calibrator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / f"{NAMESPACE}_finalist_calibrator"
FOLD_DIR = ARTIFACT_DIR / "folds"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / f"round2_{NAMESPACE}_finalist_calibrator"
)
REPORT = ANALYSIS_DIR / "fixed500_report.json"
OUTPUT_CHANNEL = ARTIFACT_DIR / "fixed500_channel.npy"
LABELS = ARTIFACT_DIR / "path_labels.npz"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
V15_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_official_sionna_path_calibrator"
    / "fixed500_report.json"
)
POINTS = 4000
CANDIDATES = 128
PERIODS = np.asarray((e85.MH, e85.MV, e85.S_SC), dtype=np.float32)
TINY = 1e-30
V15_SCORE = 0.6863750036074728
TARGET_SCORE = 0.700000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "fold", "aggregate"), default="fold")
    parser.add_argument("--outer", type=int, choices=range(5), default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(portable_payload(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def row_digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def fold_paths(outer: int) -> dict[str, Path]:
    return {
        "screen": FOLD_DIR / f"outer{outer}_screen.pt",
        "refit": FOLD_DIR / f"outer{outer}_refit.pt",
        "channel": FOLD_DIR / f"outer{outer}_channel.npy",
        "endpoint": FOLD_DIR / f"outer{outer}_endpoint.npz",
        "report": ANALYSIS_DIR / f"outer{outer}_report.json",
        "failure": ANALYSIS_DIR / f"outer{outer}_failure.json",
    }


class E200Store:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.base_source = np.load(FEATURES, mmap_mode="r")
        self.base = torch.as_tensor(
            np.asarray(self.base_source, dtype=np.float32), device=device
        )
        fraction_np = np.asarray(
            np.load(CACHE_DIR / "energy_fraction.npy", mmap_mode="r"),
            dtype=np.float32,
        )
        presence_np = np.asarray(
            np.load(CACHE_DIR / "presence.npy", mmap_mode="r"), dtype=np.float32
        )
        port_np = np.asarray(np.load(PORT_PRIOR, mmap_mode="r"), dtype=np.float32)
        self.fallback_mass = e190.normalize_rows(
            np.maximum(fraction_np, 1e-12)
            * np.sqrt(np.maximum(presence_np, 1.0))
        )
        self.fallback_port = port_np.reshape(POINTS, CANDIDATES, 8)
        self.fraction = torch.as_tensor(fraction_np, device=device)
        self.presence = torch.as_tensor(presence_np, device=device)
        self.port = torch.as_tensor(self.fallback_port, device=device)
        self.coordinate = torch.as_tensor(
            np.asarray(np.load(COORDINATE, mmap_mode="r"), dtype=np.float32),
            device=device,
        )
        self.sequence_geometry = torch.as_tensor(
            np.asarray(np.load(SEQUENCE_GEOMETRY, mmap_mode="r")), device=device
        )
        self.sequence_hash = torch.as_tensor(
            np.asarray(np.load(SEQUENCE_HASH, mmap_mode="r"), dtype=np.int64),
            device=device,
        )
        self.depth = torch.as_tensor(
            np.asarray(
                np.load(CACHE_DIR / "interaction_count.npy", mmap_mode="r"),
                dtype=np.int64,
            ),
            device=device,
        )
        self.interactions = torch.as_tensor(
            np.asarray(
                np.load(CACHE_DIR / "interactions.npy", mmap_mode="r"),
                dtype=np.int64,
            ),
            device=device,
        )
        self.objects = torch.as_tensor(
            np.asarray(
                np.load(CACHE_DIR / "objects.npy", mmap_mode="r"), dtype=np.int64
            ),
            device=device,
        )
        self.primitives = torch.as_tensor(
            np.asarray(
                np.load(CACHE_DIR / "primitives.npy", mmap_mode="r"),
                dtype=np.int64,
            ),
            device=device,
        )
        self.target_pas = torch.as_tensor(
            np.asarray(np.load(e85.TARGET_PAS, mmap_mode="r"), dtype=np.float32),
            device=device,
        )
        self.target_pdp = torch.as_tensor(
            np.asarray(np.load(e85.TARGET_PDP, mmap_mode="r"), dtype=np.float32),
            device=device,
        )
        self.power = np.asarray(np.load(e85.POINT_POWER, mmap_mode="r"), dtype=np.float64)
        self.candidate_mask = torch.as_tensor(
            np.asarray(np.load(CACHE_DIR / "mask.npy", mmap_mode="r"), dtype=bool),
            device=device,
        )

    def normalization(self, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        total = np.zeros(self.base_source.shape[-1], dtype=np.float64)
        square = np.zeros_like(total)
        count = 0
        for start in range(0, len(rows), 128):
            value = np.asarray(
                self.base_source[rows[start : start + 128]], dtype=np.float64
            )
            total += value.sum(axis=(0, 1))
            square += np.square(value).sum(axis=(0, 1))
            count += value.shape[0] * value.shape[1]
        mean = total / max(count, 1)
        variance = np.maximum(square / max(count, 1) - mean * mean, 1e-8)
        std = np.sqrt(variance)
        std[std < 1e-4] = 1.0
        return (
            torch.as_tensor(mean.astype(np.float32), device=self.device),
            torch.as_tensor(std.astype(np.float32), device=self.device),
        )

    def batch(
        self,
        rows: np.ndarray,
        mean: torch.Tensor,
        std: torch.Tensor,
        include_probe: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        del include_probe
        ids = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        anchored = self.fraction[ids].clamp_min(1e-12) * torch.sqrt(
            self.presence[ids].clamp_min(1.0)
        )
        batch = {
            "base": (self.base[ids] - mean) / std,
            "base_log_mass": torch.log(anchored.clamp_min(1e-12)),
            "coordinate": self.coordinate[ids],
            "base_log_port": torch.log(self.port[ids].clamp_min(1e-8)),
            "sequence_geometry": self.sequence_geometry[ids].float(),
            "sequence_hash": self.sequence_hash[ids],
            "depth": self.depth[ids],
            "interactions": self.interactions[ids],
            "objects": self.objects[ids],
            "primitives": self.primitives[ids],
            "candidate_mask": self.candidate_mask[ids].to(torch.float32),
        }
        return batch, self.target_pas[ids], self.target_pdp[ids]


def prepare_labels(device: torch.device, force: bool) -> dict[str, Any]:
    coordinate = np.asarray(np.load(COORDINATE, mmap_mode="r"), dtype=np.float32)
    mask = np.asarray(np.load(CACHE_DIR / "mask.npy", mmap_mode="r"), dtype=bool)
    coordinate_digest = row_digest(coordinate, mask)
    if LABELS.is_file() and not force:
        cached = np.load(LABELS, allow_pickle=False)
        if str(cached["coordinate_digest"].item()) == coordinate_digest:
            report = json.loads(str(cached["report_json"].item()))
            return {"status": "reused", "coordinate_digest": coordinate_digest, **report}
    store = E200Store(device)
    component = np.load(e190.COMPONENTS, allow_pickle=False)
    config = e190.Config()
    labels, report = e190.build_labels(
        coordinate,
        mask,
        component["coordinates"].astype(np.float32),
        component["coefficients"].astype(np.complex64),
        component["point_power"].astype(np.float64),
        store.fallback_mass,
        store.fallback_port,
        config,
    )
    atomic_npz(
        LABELS,
        mass=labels.mass,
        delta=labels.delta,
        port=labels.port,
        matched_count=labels.matched_count,
        matched_fraction=labels.matched_fraction,
        valid=labels.valid,
        coordinate_digest=np.asarray(coordinate_digest),
        report_json=np.asarray(json.dumps(report, ensure_ascii=False)),
    )
    del store
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"status": "created", "coordinate_digest": coordinate_digest, **report}


def load_labels(device: torch.device) -> e195.ResidentLabels:
    if not LABELS.is_file():
        raise FileNotFoundError(PATHS.relative(LABELS))
    return e195.ResidentLabels(LABELS, device)


def run_fold(args: argparse.Namespace) -> dict[str, Any]:
    paths = fold_paths(args.outer)
    if paths["report"].is_file() and not args.force:
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        if report.get("status") == "fold_complete":
            return report
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    phase: dict[str, float] = {}
    config = e190.Config(outer=args.outer, validation_fold=(args.outer + 1) % 5)
    try:
        load_started = time.time()
        store = E200Store(device)
        labels = load_labels(device)
        evaluator = e195.IndependentEvaluator(device)
        split = e195.canonical_split(args.outer, config.validation_fold, evaluator)
        coordinate = np.asarray(np.load(COORDINATE, mmap_mode="r"), dtype=np.float32)
        mask = np.asarray(np.load(CACHE_DIR / "mask.npy", mmap_mode="r"), dtype=bool)
        coordinate_digest = row_digest(coordinate, mask)
        phase["resident_load"] = float(time.time() - load_started)

        screen_fit = split["screen_fit"]
        validation_rows = split["validation"]
        refit_rows = split["refit"]
        outer_rows = split["outer"]
        mean_screen, std_screen = store.normalization(screen_fit)
        screen_digest = row_digest(screen_fit, validation_rows, outer_rows)
        screen_started = time.time()
        screen_checkpoint, screen_report = e195.train_strict_base_screen(
            paths["screen"],
            store,
            labels,
            screen_fit,
            validation_rows,
            mean_screen,
            std_screen,
            evaluator,
            config,
            config.batch_size,
            screen_digest,
            coordinate_digest,
        )
        phase["screen_training"] = float(time.time() - screen_started)
        best_epoch = int(screen_checkpoint["best_epoch"])
        del screen_checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

        mean_refit, std_refit = store.normalization(refit_rows)
        refit_digest = row_digest(refit_rows, outer_rows)
        refit_started = time.time()
        refit_checkpoint, refit_report = e195.train_strict_base_refit(
            paths["refit"],
            store,
            labels,
            refit_rows,
            mean_refit,
            std_refit,
            best_epoch,
            config,
            refit_digest,
            coordinate_digest,
        )
        model = e195.make_fresh_base_model(
            store, config, device, config.random_state + 100000
        )
        model.load_state_dict(refit_checkpoint["state_dict"])
        mass, endpoint_coordinate, port = e195.endpoint_arrays(
            model,
            store,
            outer_rows,
            mean_refit,
            std_refit,
            config.batch_size,
            include_probe=False,
        )
        endpoint_pas, endpoint_pdp = e190.render_endpoint(
            mass, endpoint_coordinate, port, device
        )
        selected, channel = evaluator.evaluate(
            outer_rows, endpoint_pas, endpoint_pdp, collect_channel=True
        )
        if channel is None:
            raise RuntimeError("E200 outer channel missing")
        phase["refit_training_and_outer"] = float(time.time() - refit_started)
        v15 = json.loads(V15_REPORT.read_text(encoding="utf-8"))
        v15_fold = next(row for row in v15["folds"] if int(row["outer"]) == args.outer)
        delta = e190.metric_delta(selected, v15_fold["selected"])
        atomic_npy(paths["channel"], channel)
        atomic_npz(
            paths["endpoint"],
            query_ids=outer_rows.astype(np.int32),
            mass=mass,
            coordinate=endpoint_coordinate,
            port=port,
        )
        peak_memory = (
            float(torch.cuda.max_memory_allocated() / 2**30)
            if device.type == "cuda"
            else 0.0
        )
        report = {
            "experiment": VERSION,
            "status": "fold_complete",
            "configuration": asdict(config),
            "outer": args.outer,
            "validation_fold": config.validation_fold,
            "split": {
                "screen_fit": int(len(screen_fit)),
                "validation": int(len(validation_rows)),
                "refit": int(len(refit_rows)),
                "outer": int(len(outer_rows)),
                "zero_label_overlap": split["leakage"],
                "screen_digest": screen_digest,
                "refit_digest": refit_digest,
            },
            "training": {
                "best_epoch": best_epoch,
                "screen": screen_report,
                "refit": refit_report,
            },
            "selected": selected,
            "v15": v15_fold["selected"],
            "delta_vs_v15": delta,
            "data_lineage": {
                "historical_v7_v15_prediction_h_read": False,
                "historical_v15_checkpoint_or_endpoint_read": False,
                "official_train_h_used_for_labels": True,
                "same_position_external_sionna_paths_used": True,
                "independent_v11_equivalent_base": PATHS.relative(e195.INDEPENDENT_BASE),
                "candidate_cache_report": PATHS.relative(CACHE_REPORT),
            },
            "output": {
                "channel": PATHS.relative(paths["channel"]),
                "endpoint": PATHS.relative(paths["endpoint"]),
                "shape": list(channel.shape),
                "dtype": str(channel.dtype),
                "finite": bool(np.isfinite(channel).all()),
            },
            "performance": {
                "device": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu",
                "peak_torch_memory_gib": peak_memory,
                "peak_rss_gib": float(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
                ),
                "phase_seconds": phase,
                "wall_seconds": float(time.time() - started),
                "measured_bottleneck": max(phase, key=phase.get),
            },
        }
        atomic_json(paths["report"], report)
        return report
    except BaseException:
        failure = {
            "experiment": VERSION,
            "status": "failed",
            "outer": args.outer,
            "traceback": traceback.format_exc(),
            "wall_seconds": float(time.time() - started),
        }
        atomic_json(paths["failure"], failure)
        raise


def aggregate(device: torch.device) -> dict[str, Any]:
    started = time.time()
    evaluator = e195.IndependentEvaluator(device)
    channel = np.zeros((len(evaluator.query_ids), 256, 4, 192), dtype=np.complex64)
    folds = []
    for outer in range(5):
        paths = fold_paths(outer)
        required = (paths["report"], paths["channel"], paths["endpoint"])
        missing = [PATHS.relative(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing E200 fold outputs: {missing}")
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        if report.get("status") != "fold_complete":
            raise RuntimeError(f"E200 outer{outer} incomplete")
        endpoint = np.load(paths["endpoint"], allow_pickle=False)
        query_ids = endpoint["query_ids"].astype(np.int64)
        expected = evaluator.query_ids[evaluator.fold_id == outer]
        if not np.array_equal(query_ids, expected):
            raise RuntimeError(f"E200 outer{outer} query identity mismatch")
        local = evaluator.lookup[query_ids]
        local_channel = np.load(paths["channel"], mmap_mode="r")
        if local_channel.shape != (len(query_ids), 256, 4, 192):
            raise RuntimeError(f"E200 outer{outer} channel shape mismatch")
        channel[local] = np.asarray(local_channel, dtype=np.complex64)
        folds.append(
            {
                "outer": outer,
                "count": int(len(query_ids)),
                "selected": report["selected"],
                "v15": report["v15"],
                "delta_vs_v15": report["delta_vs_v15"],
                "best_epoch": int(report["training"]["best_epoch"]),
            }
        )
    selected = evaluator.score_channel(channel)
    v15 = json.loads(V15_REPORT.read_text(encoding="utf-8"))["selected"]
    delta = e190.metric_delta(selected, v15)
    positive_folds = int(sum(row["delta_vs_v15"]["score"] > 0.0 for row in folds))
    worst_fold_delta = float(min(row["delta_vs_v15"]["score"] for row in folds))
    passed = bool(float(selected["score"]) > V15_SCORE)
    target_passed = bool(
        float(selected["score"]) >= TARGET_SCORE
        and positive_folds >= 4
        and worst_fold_delta >= -0.001
    )
    atomic_npy(OUTPUT_CHANNEL, channel)
    report = {
        "experiment": VERSION,
        "status": "fixed500_passed" if passed else "fixed500_failed",
        "selected": selected,
        "v15": v15,
        "delta_vs_v15": delta,
        "delta_vs_target": float(selected["score"] - TARGET_SCORE),
        "folds": folds,
        "gate": {
            "minimum_score": V15_SCORE,
            "target_score": TARGET_SCORE,
            "positive_folds": positive_folds,
            "worst_fold_delta": worst_fold_delta,
            "v15_beaten": passed,
            "large_gain_target_passed": target_passed,
        },
        "output": {
            "channel": PATHS.relative(OUTPUT_CHANNEL),
            "shape": list(channel.shape),
            "dtype": str(channel.dtype),
            "finite": bool(np.isfinite(channel).all()),
        },
        "data_lineage": {
            "historical_v7_v15_prediction_h_read": False,
            "historical_v15_checkpoint_or_endpoint_read": False,
            "official_test_h_used": False,
            "same_position_test_sionna_generated": False,
        },
        "performance": {
            "device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "wall_seconds": float(time.time() - started),
            "measured_bottleneck": "official full-channel aggregation and scoring",
        },
    }
    atomic_json(REPORT, report)
    return report


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.stage == "prepare":
        report = prepare_labels(device, args.force)
    elif args.stage == "fold":
        report = run_fold(args)
    else:
        report = aggregate(device)
    print(json.dumps(portable_payload(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
