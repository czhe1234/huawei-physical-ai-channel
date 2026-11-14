#!/usr/bin/env python3
"""Run the complete Physical-AI submission pipeline from official inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from project_paths import PATHS

from .config import PipelineConfig, RuntimePaths
from .io_utils import atomic_json, validate_channel


PHASES = (
    "trace",
    "spectra",
    "tokens",
    "base",
    "state",
    "predict",
    "all",
    "train-trace",
    "train-spectra",
    "train-calibrators",
    "train-spectral-all",
    "train-tokens",
    "train-path-state",
    "train-donor",
    "train-carrier-assets",
    "train-spectral-assets",
    "train-lifecycle",
    "build-oof-endpoints",
    "build-oof-joint-field",
    "train-link-gate",
    "evaluate-retrained",
    "verify-retrained-assets",
    "train-all",
    "validate",
)

RETRAINED_PHASES = frozenset(
    phase
    for phase in PHASES
    if phase.startswith("train-")
    or phase.startswith("build-oof-")
    or phase in ("evaluate-retrained", "verify-retrained-assets")
)

REQUIRED_DATA_FILES = (
    "Round2_Train_Channel.npy",
    "Round2_Train_Pos.npy",
    "Round2_Test_Pos.npy",
    "Round2_Map.ply",
    "Round2_Setup.json",
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
    root = PATHS.root
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        candidates = (root / "datasets" / "round2", root.parent / "datasets" / "round2")
        data_dir = next(
            (
                candidate
                for candidate in candidates
                if all((candidate / name).is_file() for name in REQUIRED_DATA_FILES)
            ),
            candidates[0],
        )
    use_retrained = args.retrained or args.phase in RETRAINED_PHASES
    default_work = (
        root / "artifacts" / "round2" / "physical_ai_retrained"
        if use_retrained
        else None
    )
    work_dir = Path(args.work_dir) if args.work_dir else default_work
    model_dir = (
        Path(args.model_dir)
        if args.model_dir
        else (work_dir / "models" if use_retrained and work_dir is not None else None)
    )
    output = (
        Path(args.output)
        if args.output
        else (root / "Round2_Test_Channel_retrained.npy" if use_retrained else None)
    )
    return RuntimePaths.create(
        project_root=root,
        data_dir=data_dir,
        work_dir=work_dir,
        model_dir=model_dir,
        output=output,
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


def _run_train_trace(
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


def _run_train_tokens(
    paths: RuntimePaths,
    config: PipelineConfig,
    force: bool,
) -> dict[str, Any]:
    from .features import build_model_inputs
    from .path_tokens import build_path_tokens

    tokens = build_path_tokens(paths, config, "train", force=force)
    inputs = build_model_inputs(paths, config, "train", force=force)
    return {"tokens": tokens, "model_inputs": inputs}


def _run_train_path_state(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    preparation_batch_size: int,
    path_batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from .path_training import PathTrainingConfig, train_path_state_models

    return train_path_state_models(
        paths,
        config,
        device,
        preparation_batch_size,
        force=force,
        training_config=PathTrainingConfig(batch_size=path_batch_size),
    )


def _run_train_donor(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    preparation_batch_size: int,
    donor_batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from .donor_training import DonorTrainingConfig, train_donor_model

    return train_donor_model(
        paths,
        config,
        device,
        preparation_batch_size,
        force=force,
        training_config=DonorTrainingConfig(batch_size=donor_batch_size),
    )


def _run_train_carrier_assets(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from .carrier_training import train_carrier_assets

    return train_carrier_assets(paths, config, device, batch_size, force=force)


def _run_train_spectral_assets(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from .spectral_assets import train_spectral_assets

    return train_spectral_assets(paths, config, device, batch_size, force=force)


def _run_train_lifecycle(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from .lifecycle_training import train_lifecycle_policy

    return train_lifecycle_policy(paths, config, device, batch_size, force=force)


def _run_oof_endpoints(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from .oof_base import build_oof_spectral_endpoints

    return build_oof_spectral_endpoints(paths, config, device, batch_size, force=force)


def _run_oof_joint_field(
    paths: RuntimePaths,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from .oof_base import build_oof_joint_field

    return build_oof_joint_field(paths, device, batch_size, force=force)


def _run_train_link_gate(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    from .link_training import train_link_gate_models

    return train_link_gate_models(paths, config, device, batch_size, force=force)


def _run_evaluate_retrained(
    paths: RuntimePaths,
    config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool,
    success_threshold: float,
) -> dict[str, Any]:
    from .evaluation import evaluate_retrained

    return evaluate_retrained(
        paths,
        config,
        device,
        batch_size,
        force=force,
        success_threshold=success_threshold,
    )


def _run_inventory(paths: RuntimePaths, require_complete: bool = True) -> dict[str, Any]:
    from .inventory import validate_production_inventory

    return validate_production_inventory(paths, require_complete=require_complete)


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


def _run_train_all(
    paths: RuntimePaths,
    config: PipelineConfig,
    gpu_ids: tuple[int, ...],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from .training_common import write_failure_report

    failure_path = paths.report_dir / "train_all_failure.json"
    failure_path.unlink(missing_ok=True)
    stages = (
        (
            "train_trace",
            lambda: _run_train_trace(
                paths, config, gpu_ids, args.force, args.workers_per_gpu
            ),
        ),
        (
            "train_spectra",
            lambda: _run_train_spectra(
                paths, config, gpu_ids, args.force, args.workers_per_gpu
            ),
        ),
        ("train_tokens", lambda: _run_train_tokens(paths, config, args.force)),
        (
            "train_calibrators",
            lambda: _run_train_calibrators(
                paths,
                config,
                device,
                args.batch_size,
                args.train_domains,
                args.train_folds,
                args.train_components,
                args.force,
            ),
        ),
        (
            "train_path_state",
            lambda: _run_train_path_state(
                paths,
                config,
                device,
                args.batch_size,
                args.path_batch_size,
                args.force,
            ),
        ),
        (
            "train_donor",
            lambda: _run_train_donor(
                paths,
                config,
                device,
                args.batch_size,
                args.donor_batch_size,
                args.force,
            ),
        ),
        (
            "train_carrier_assets",
            lambda: _run_train_carrier_assets(
                paths, config, device, args.batch_size, args.force
            ),
        ),
        (
            "train_spectral_assets",
            lambda: _run_train_spectral_assets(
                paths, config, device, args.batch_size, args.force
            ),
        ),
        (
            "train_lifecycle",
            lambda: _run_train_lifecycle(
                paths, config, device, args.batch_size, args.force
            ),
        ),
        (
            "build_oof_endpoints",
            lambda: _run_oof_endpoints(
                paths, config, device, args.batch_size, args.force
            ),
        ),
        (
            "build_oof_joint_field",
            lambda: _run_oof_joint_field(
                paths, device, args.batch_size, args.force
            ),
        ),
        (
            "train_link_gate",
            lambda: _run_train_link_gate(
                paths, config, device, args.batch_size, args.force
            ),
        ),
        (
            "evaluate_retrained",
            lambda: _run_evaluate_retrained(
                paths,
                config,
                device,
                args.batch_size,
                args.force,
                args.success_threshold,
            ),
        ),
        ("verify_retrained_assets", lambda: _run_inventory(paths)),
    )
    summaries: dict[str, Any] = {}
    completed: list[str] = []
    progress_path = paths.report_dir / "train_all_progress.json"
    for name, runner in stages:
        atomic_json(
            progress_path,
            {
                "stage": "train_all",
                "status": "running",
                "active_stage": name,
                "completed_stages": completed,
            },
        )
        started = time.time()
        try:
            report = runner()
        except BaseException as error:
            write_failure_report(
                failure_path, name, error
            )
            raise
        elapsed = float(time.time() - started)
        summary: dict[str, Any] = {
            "status": report.get("status", "complete"),
            "wall_seconds": elapsed,
        }
        if name == "evaluate_retrained":
            summary["metrics"] = report["metrics"]
            summary["quality_gate_passed"] = report["quality_gate_passed"]
        if name == "verify_retrained_assets":
            summary["valid_checkpoint_count"] = report["valid_checkpoint_count"]
            summary["valid_asset_count"] = report["valid_asset_count"]
        summaries[name] = summary
        completed.append(name)
    atomic_json(
        progress_path,
        {
            "stage": "train_all",
            "status": "complete",
            "active_stage": None,
            "completed_stages": completed,
            "summaries": summaries,
        },
    )
    return {"stage": "train_all", "status": "complete", "stages": summaries}


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
    parser.add_argument("--path-batch-size", type=int, default=32)
    parser.add_argument("--donor-batch-size", type=int, default=128)
    parser.add_argument("--train-domains", default="pas,pdp")
    parser.add_argument("--train-folds", default="0,1,2,3,4")
    parser.add_argument("--train-components", default="spectral,joint")
    parser.add_argument("--success-threshold", type=float, default=0.670000)
    parser.add_argument(
        "--retrained",
        action="store_true",
        help="use the isolated physical_ai_retrained work/model/output defaults",
    )
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
    if args.phase == "verify-retrained-assets":
        print(
            json.dumps(
                {"verify_retrained_assets": _run_inventory(paths)},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    try:
        phase_failure_path = (
            paths.report_dir / f"{args.phase.replace('-', '_')}_failure.json"
        )
        phase_failure_path.unlink(missing_ok=True)
        gpu_ids = _gpu_ids(args.gpus)
        device = _device(args.device, gpu_ids)
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
            reports["base"] = _run_base(
                paths, config, device, args.batch_size, args.force
            )
        if args.phase in ("state", "all"):
            reports["state"] = _run_state(paths, config, device, args.batch_size)
        if args.phase in ("predict", "all"):
            reports["predict"] = _run_predict(paths, config, device, args.batch_size)
        if args.phase in ("train-trace", "train-spectral-all"):
            reports["train_trace"] = _run_train_trace(
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
        if args.phase == "train-tokens":
            reports["train_tokens"] = _run_train_tokens(paths, config, args.force)
        if args.phase == "train-path-state":
            reports["train_path_state"] = _run_train_path_state(
                paths,
                config,
                device,
                args.batch_size,
                args.path_batch_size,
                args.force,
            )
        if args.phase == "train-donor":
            reports["train_donor"] = _run_train_donor(
                paths,
                config,
                device,
                args.batch_size,
                args.donor_batch_size,
                args.force,
            )
        if args.phase == "train-carrier-assets":
            reports["train_carrier_assets"] = _run_train_carrier_assets(
                paths, config, device, args.batch_size, args.force
            )
        if args.phase == "train-spectral-assets":
            reports["train_spectral_assets"] = _run_train_spectral_assets(
                paths, config, device, args.batch_size, args.force
            )
        if args.phase == "train-lifecycle":
            reports["train_lifecycle"] = _run_train_lifecycle(
                paths, config, device, args.batch_size, args.force
            )
        if args.phase == "build-oof-endpoints":
            reports["build_oof_endpoints"] = _run_oof_endpoints(
                paths, config, device, args.batch_size, args.force
            )
        if args.phase == "build-oof-joint-field":
            reports["build_oof_joint_field"] = _run_oof_joint_field(
                paths, device, args.batch_size, args.force
            )
        if args.phase == "train-link-gate":
            reports["train_link_gate"] = _run_train_link_gate(
                paths, config, device, args.batch_size, args.force
            )
        if args.phase == "evaluate-retrained":
            reports["evaluate_retrained"] = _run_evaluate_retrained(
                paths,
                config,
                device,
                args.batch_size,
                args.force,
                args.success_threshold,
            )
        if args.phase == "train-all":
            reports["train_all"] = _run_train_all(
                paths, config, gpu_ids, device, args
            )
    except BaseException as error:
        from .training_common import write_failure_report

        write_failure_report(
            phase_failure_path,
            args.phase,
            error,
        )
        raise
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
