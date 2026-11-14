from typing import Any

import torch

from ..config import PipelineConfig, RuntimePaths


def run_prepare_prior(
    paths: RuntimePaths,
    config: PipelineConfig,
    gpu_ids: tuple[int, ...],
    force: bool,
    workers_per_gpu: int,
) -> dict[str, Any]:
    from ..scene_prior.support_preparation import prepare_prior_supports

    return prepare_prior_supports(
        paths,
        config,
        "test",
        gpu_ids,
        force=force,
        workers_per_gpu=workers_per_gpu,
    )


def run_spectra(
    paths: RuntimePaths,
    config: PipelineConfig,
    gpu_ids: tuple[int, ...],
    force: bool,
    workers_per_gpu: int,
) -> dict[str, Any]:
    from ..scene_prior.spectral_statistics import (
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


def run_tokens(
    paths: RuntimePaths,
    config: PipelineConfig,
    force: bool,
) -> dict[str, Any]:
    from ..scene_prior.component_aggregation import build_component_tokens
    from ..scene_prior.component_features import build_model_inputs

    tokens = build_component_tokens(paths, config, "test", force=force)
    inputs = build_model_inputs(paths, config, "test", force=force)
    return {"tokens": tokens, "model_inputs": inputs}


def run_base(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from ..ai_enhancement.base_channel_builder import build_base_field

    return build_base_field(paths, config, device, batch_size, force=force)
