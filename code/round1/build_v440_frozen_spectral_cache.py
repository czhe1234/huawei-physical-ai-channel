#!/usr/bin/env python3
"""Cache frozen v439 spectral and component predictions for v440 training."""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import train_v439_component_route_long_model as v439
from project_paths import PATHS


VERSION_NUMBER = 440
VERSION = "v440_frozen_spectral_cache"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CHECKPOINT = (
    PATHS.checkpoints
    / "v439_component_route_long_model"
    / "component_route_best.pt"
)

COUNT = 100_000
BATCH_SIZE = 192


def cache_path(name: str):
    return OUT_DIR / f"{name}.npy"


def create_memmap(name: str, shape: tuple[int, ...]) -> np.memmap:
    final = cache_path(name)
    partial = final.with_name(final.stem + ".partial.npy")
    if partial.exists():
        partial.unlink()
    return np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float16, shape=shape
    )


def finish_memmap(name: str, value: np.memmap) -> None:
    value.flush()
    del value
    final = cache_path(name)
    partial = final.with_name(final.stem + ".partial.npy")
    partial.replace(final)


def validate() -> dict[str, object]:
    expected = {
        "base_joint": (COUNT, 192, 16),
        "path_delay": (COUNT, 192),
        "component_delay": (COUNT, 32),
        "component_h": (COUNT, 16),
        "component_mass": (COUNT, 82),
    }
    files = {}
    for name, shape in expected.items():
        path = cache_path(name)
        value = np.load(path, mmap_mode="r")
        if value.shape != shape or value.dtype != np.float16:
            raise ValueError(f"{name}: {value.shape}/{value.dtype}, expected {shape}/float16")
        flat = value.reshape(-1)
        index = np.linspace(0, len(flat) - 1, min(100_000, len(flat)), dtype=np.int64)
        if not np.isfinite(flat[index]).all():
            raise FloatingPointError(f"{name} contains non-finite sampled values")
        files[name] = {"shape": list(shape), "bytes": int(path.stat().st_size)}
    return {
        "status": "pass",
        "files": files,
        "bytes": int(sum(row["bytes"] for row in files.values())),
    }


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    names = {
        "base_joint": (COUNT, 192, 16),
        "path_delay": (COUNT, 192),
        "component_delay": (COUNT, 32),
        "component_h": (COUNT, 16),
        "component_mass": (COUNT, 82),
    }
    if all(cache_path(name).is_file() for name in names):
        status = "reused"
        seconds = 0.0
    else:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"v440 cache requires exactly one visible GPU, got {torch.cuda.device_count()}"
            )
        device = torch.device("cuda")
        data = v439.TrainingData()
        model = v439.ComponentRouteTransformer(0.0).to(device)
        checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        buffers = {name: create_memmap(name, shape) for name, shape in names.items()}
        started = time.time()
        with torch.no_grad():
            for start in range(0, COUNT, BATCH_SIZE):
                end = min(start + BATCH_SIZE, COUNT)
                ids = np.arange(start, end, dtype=np.int64)
                batch = data.batch(ids, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(
                        batch["route"],
                        batch["position"],
                        batch["component_feature"],
                    )
                joint = F.softmax(output["joint_logits"].flatten(1), dim=1).reshape(
                    -1, 192, 16
                )
                path_delay = F.softmax(output["path_delay_logits"], dim=1)
                component_delay = F.softmax(
                    output["component_delay_logits"].flatten(1), dim=1
                ).reshape(-1, 82, 32).sum(dim=1)
                component_h = F.softmax(
                    output["component_h_logits"].flatten(1), dim=1
                ).reshape(-1, 82, 16).sum(dim=1)
                component_mass = F.softmax(output["component_mass_logits"], dim=1)
                for name, value in (
                    ("base_joint", joint),
                    ("path_delay", path_delay),
                    ("component_delay", component_delay),
                    ("component_h", component_h),
                    ("component_mass", component_mass),
                ):
                    buffers[name][start:end] = value.float().cpu().numpy().astype(np.float16)
                print(f"v440 frozen predictions {end}/{COUNT}", flush=True)
        for name in tuple(buffers):
            finish_memmap(name, buffers.pop(name))
        seconds = float(time.time() - started)
        status = "generated"
    report = {
        "method": VERSION,
        "status": "frozen_prediction_cache_ready",
        "source_checkpoint": PATHS.relative(CHECKPOINT),
        "generation": {"status": status, "seconds": seconds},
        "validation": validate(),
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
