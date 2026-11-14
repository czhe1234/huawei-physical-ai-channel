#!/usr/bin/env python3
"""Run the complete Physical-AI submission pipeline from official inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from project_paths import PATHS

from .config import PipelineConfig, RuntimePaths
from .io_utils import validate_channel


PHASES = ("trace", "spectra", "tokens", "base", "state", "predict", "all", "validate")


def _gpu_ids(value: str | None) -> tuple[int, ...]:
    if value:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        result = tuple(range(torch.cuda.device_count()))
    if not result:
        raise RuntimeError("at least one CUDA GPU is required")
    return result


def _device(value: str | None, gpu_ids: tuple[int, ...]) -> torch.device:
    return torch.device(value or f"cuda:{gpu_ids[0]}")


def _paths(args: argparse.Namespace) -> RuntimePaths:
    return RuntimePaths.create(
        project_root=PATHS.root,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        output=Path(args.output) if args.output else None,
    )


def _run_trace(
    paths: RuntimePaths,
    config: PipelineConfig,
    gpu_ids: tuple[int, ...],
    force: bool,
    workers_per_gpu: int,
) -> dict[str, Any]:
    from .propagation import run_propagation

    return run_propagation(
        paths,
        config,
        "test",
        gpu_ids,
        force=force,
        workers_per_gpu=workers_per_gpu,
    )


def _run_spectra(
    paths: RuntimePaths,
    config: PipelineConfig,
    gpu_ids: tuple[int, ...],
    force: bool,
    workers_per_gpu: int,
) -> dict[str, Any]:
    from .physical_spectra import (
        aggregate_spectral_statistics,
        build_spectral_shards,
    )

    shards = build_spectral_shards(
        paths,
        config,
        "test",
        gpu_ids,
        force=force,
        workers_per_gpu=workers_per_gpu,
    )
    aggregate = aggregate_spectral_statistics(paths, config, "test", force=force)
    return {"shards": shards, "aggregate": aggregate}


def _run_tokens(
    paths: RuntimePaths,
    config: PipelineConfig,
    force: bool,
) -> dict[str, Any]:
    from .features import build_model_inputs
    from .path_tokens import build_path_tokens

    tokens = build_path_tokens(paths, config, "test", force=force)
    inputs = build_model_inputs(paths, config, "test", force=force)
    return {"tokens": tokens, "model_inputs": inputs}


def _run_base(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from .base_field import build_base_field

    return build_base_field(paths, config, device, batch_size, force=force)


def _run_state(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    from .model import predict_path_states

    return predict_path_states(paths, config, "test", device, batch_size=batch_size)


def _run_predict(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    from .renderer import synthesize_channel

    return synthesize_channel(
        paths,
        config,
        "test",
        paths.base_dir / "base_channel.npy",
        paths.output,
        device,
        batch_size=batch_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--data-dir")
    parser.add_argument("--work-dir")
    parser.add_argument("--model-dir")
    parser.add_argument("--output")
    parser.add_argument("--gpus", help="comma-separated CUDA device indices")
    parser.add_argument("--device", help="PyTorch device for AI/FFT stages")
    parser.add_argument("--seed-count", type=int, default=24)
    parser.add_argument("--receiver-batch-size", type=int, default=16)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_ids = _gpu_ids(args.gpus)
    device = _device(args.device, gpu_ids)
    paths = _paths(args)
    config = PipelineConfig(
        seed_count=args.seed_count,
        receiver_batch_size=args.receiver_batch_size,
    )
    reports: dict[str, Any] = {}

    if args.phase in ("trace", "all"):
        reports["trace"] = _run_trace(
            paths, config, gpu_ids, args.force, args.workers_per_gpu
        )
    if args.phase in ("spectra", "all"):
        reports["spectra"] = _run_spectra(
            paths,
            config,
            gpu_ids,
            args.force,
            args.workers_per_gpu,
        )
    if args.phase in ("tokens", "all"):
        reports["tokens"] = _run_tokens(paths, config, args.force)
    if args.phase in ("base", "all"):
        reports["base"] = _run_base(paths, config, device, args.batch_size, args.force)
    if args.phase in ("state", "all"):
        reports["state"] = _run_state(paths, config, device, args.batch_size)
    if args.phase in ("predict", "all"):
        reports["predict"] = _run_predict(paths, config, device, args.batch_size)
    if args.phase == "validate":
        reports["validate"] = validate_channel(paths.output, 500)

    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
