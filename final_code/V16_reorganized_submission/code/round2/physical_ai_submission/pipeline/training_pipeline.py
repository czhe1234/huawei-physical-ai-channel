from typing import Any

import torch

from ..config import PipelineConfig, RuntimePaths


def run_prepare_train_prior(
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
        "train",
        gpu_ids,
        force=force,
        workers_per_gpu=workers_per_gpu,
    )


def run_train_spectra(
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
        "train",
        gpu_ids,
        force=force,
        workers_per_gpu=workers_per_gpu,
    )
    aggregate = aggregate_spectral_statistics(paths, config, "train", force=force)
    return {"shards": shards, "aggregate": aggregate}


def run_train_calibrators(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    domains_value: str,
    folds_value: str,
    components_value: str,
    force: bool,
) -> dict[str, Any]:
    from ..training.spectral_models import (
        COMPONENTS,
        DOMAINS,
        parse_folds,
        parse_selection,
        train_spectral_calibrators,
    )

    domains = parse_selection(domains_value, DOMAINS, "train domains")
    components = parse_selection(components_value, COMPONENTS, "train components")
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


def run_train_tokens(
    paths: RuntimePaths,
    config: PipelineConfig,
    force: bool,
) -> dict[str, Any]:
    from ..scene_prior.component_aggregation import build_component_tokens
    from ..scene_prior.component_features import build_model_inputs

    tokens = build_component_tokens(paths, config, "train", force=force)
    inputs = build_model_inputs(paths, config, "train", force=force)
    return {"tokens": tokens, "model_inputs": inputs}


def run_train_component_state(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    preparation_batch_size: int,
    component_batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from ..training.component_state import ComponentTrainingConfig, train_component_state_models

    return train_component_state_models(
        paths,
        config,
        device,
        preparation_batch_size,
        force=force,
        training_config=ComponentTrainingConfig(batch_size=component_batch_size),
    )


def run_train_donor(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    preparation_batch_size: int,
    donor_batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from ..training.donor import DonorTrainingConfig, train_donor_model

    return train_donor_model(
        paths,
        config,
        device,
        preparation_batch_size,
        force=force,
        training_config=DonorTrainingConfig(batch_size=donor_batch_size),
    )


def run_train_carrier_assets(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from ..training.carrier import train_carrier_assets

    return train_carrier_assets(paths, config, device, batch_size, force=force)


def run_train_spectral_assets(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from ..training.spectral_assets import train_spectral_assets

    return train_spectral_assets(paths, config, device, batch_size, force=force)


def run_train_lifecycle(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from ..training.lifecycle import train_lifecycle_policy

    return train_lifecycle_policy(paths, config, device, batch_size, force=force)


def run_build_oof_endpoints(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from ..training.oof_base import build_oof_spectral_endpoints

    return build_oof_spectral_endpoints(paths, config, device, batch_size, force=force)


def run_build_oof_joint_field(
    paths: RuntimePaths,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from ..training.oof_base import build_oof_joint_field

    return build_oof_joint_field(paths, device, batch_size, force=force)


def run_train_link_gate(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from ..training.link_gate import train_link_gate_models

    return train_link_gate_models(paths, config, device, batch_size, force=force)
