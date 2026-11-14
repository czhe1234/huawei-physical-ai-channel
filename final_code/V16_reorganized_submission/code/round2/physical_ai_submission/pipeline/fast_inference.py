from typing import Any

import torch

from ..config import PipelineConfig, RuntimePaths


def run_state(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    from ..component_state.model import predict_component_states

    return predict_component_states(paths, config, "test", device, batch_size=batch_size)


def run_prediction(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    from ..synthesis.channel_generator import synthesize_channel

    return synthesize_channel(
        paths,
        config,
        "test",
        paths.base_dir / "base_channel.npy",
        paths.output,
        device,
        batch_size=batch_size,
    )


def run_fast_inference(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    from ..deployment.bundle_validation import validate_deployment_bundle

    deployment = validate_deployment_bundle(paths)
    state = run_state(paths, config, device, batch_size)
    prediction = run_prediction(paths, config, device, batch_size)
    return {
        "deployment": deployment,
        "state": state,
        "predict": prediction,
    }
