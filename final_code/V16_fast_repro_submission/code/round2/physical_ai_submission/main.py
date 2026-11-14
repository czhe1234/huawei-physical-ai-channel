#!/usr/bin/env python3
"""Run fast inference or rebuild the offline Physical-AI prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from project_paths import PATHS

from .config import PipelineConfig, RuntimePaths
from .io_utils import validate_channel


PHASES = (
    "infer",
    "prepare-prior",
    "spectra",
    "tokens",
    "base",
    "state",
    "predict",
    "rebuild-all",
    "prepare-train-prior",
    "train-spectra",
    "train-calibrators",
    "train-spectral-all",
    "validate",
)


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


def _run_prepare_prior(
    paths: RuntimePaths,
    config: PipelineConfig,
    gpu_ids: tuple[int, ...],
    force: bool,
    workers_per_gpu: int,
) -> dict[str, Any]:
    from .prior_preparation import prepare_prior_supports

    return prepare_prior_supports(
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


def _run_prepare_train_prior(
    paths: RuntimePaths,
    config: PipelineConfig,
    gpu_ids: tuple[int, ...],
    force: bool,
    workers_per_gpu: int,
) -> dict[str, Any]:
    from .prior_preparation import prepare_prior_supports

    return prepare_prior_supports(
        paths,
        config,
        "train",
        gpu_ids,
        force=force,
        workers_per_gpu=workers_per_gpu,
    )


def _run_train_spectra(
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
        "train",
        gpu_ids,
        force=force,
        workers_per_gpu=workers_per_gpu,
    )
    aggregate = aggregate_spectral_statistics(paths, config, "train", force=force)
    return {"shards": shards, "aggregate": aggregate}


def _run_train_calibrators(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    domains_value: str,
    folds_value: str,
    components_value: str,
    force: bool,
) -> dict[str, Any]:
    from .training import (
        COMPONENTS,
        DOMAINS,
        parse_folds,
        parse_selection,
        train_spectral_calibrators,
    )

    domains = parse_selection(domains_value, DOMAINS, "train domains")
    components = parse_selection(
        components_value, COMPONENTS, "train components"
    )
    folds = parse_folds(folds_value)
    return train_spectral_calibrators(
        paths,
        config,
        device,
        batch_size,
        domains=domains,
        folds=folds,
        components=components,
        force=force,
    )


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


def _run_infer(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    from .deployment import validate_deployment_bundle

    deployment = validate_deployment_bundle(paths)
    state = _run_state(paths, config, device, batch_size)
    prediction = _run_predict(paths, config, device, batch_size)
    return {
        "deployment": deployment,
        "state": state,
        "predict": prediction,
    }


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
    parser.add_argument("--train-domains", default="pas,pdp")
    parser.add_argument("--train-folds", default="0,1,2,3,4")
    parser.add_argument("--train-components", default="spectral,joint")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _paths(args)
    if args.phase == "validate":
        print(
            json.dumps(
                {"validate": validate_channel(paths.output, 500)},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    gpu_ids = _gpu_ids(args.gpus)
    device = _device(args.device, gpu_ids)
    config = PipelineConfig(
        seed_count=args.seed_count,
        receiver_batch_size=args.receiver_batch_size,
    )
    reports: dict[str, Any] = {}

    if args.phase == "infer":
        reports["infer"] = _run_infer(
            paths, config, device, args.batch_size
        )
    if args.phase in ("prepare-prior", "rebuild-all"):
        reports["prepare_prior"] = _run_prepare_prior(
            paths, config, gpu_ids, args.force, args.workers_per_gpu
        )
    if args.phase in ("spectra", "rebuild-all"):
        reports["spectra"] = _run_spectra(
            paths,
            config,
            gpu_ids,
            args.force,
            args.workers_per_gpu,
        )
    if args.phase in ("tokens", "rebuild-all"):
        reports["tokens"] = _run_tokens(paths, config, args.force)
    if args.phase in ("base", "rebuild-all"):
        reports["base"] = _run_base(paths, config, device, args.batch_size, args.force)
    if args.phase in ("state", "rebuild-all"):
        reports["state"] = _run_state(paths, config, device, args.batch_size)
    if args.phase in ("predict", "rebuild-all"):
        reports["predict"] = _run_predict(paths, config, device, args.batch_size)
    if args.phase in ("prepare-train-prior", "train-spectral-all"):
        reports["prepare_train_prior"] = _run_prepare_train_prior(
            paths, config, gpu_ids, args.force, args.workers_per_gpu
        )
    if args.phase in ("train-spectra", "train-spectral-all"):
        reports["train_spectra"] = _run_train_spectra(
            paths,
            config,
            gpu_ids,
            args.force,
            args.workers_per_gpu,
        )
    if args.phase in ("train-calibrators", "train-spectral-all"):
        reports["train_calibrators"] = _run_train_calibrators(
            paths,
            config,
            device,
            args.batch_size,
            args.train_domains,
            args.train_folds,
            args.train_components,
            args.force,
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
