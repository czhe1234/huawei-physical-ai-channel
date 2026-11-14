from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from project_paths import PATHS

from .config import PipelineConfig, RuntimePaths
from .pipeline.fast_inference import (
    run_fast_inference as _run_infer,
    run_prediction as _run_predict,
    run_state as _run_state,
)
from .pipeline.offline_asset_build import (
    run_base as _run_base,
    run_prepare_prior as _run_prepare_prior,
    run_spectra as _run_spectra,
    run_tokens as _run_tokens,
)
from .pipeline.training_pipeline import (
    run_build_oof_endpoints as _run_build_oof_endpoints,
    run_build_oof_joint_field as _run_build_oof_joint_field,
    run_prepare_train_prior as _run_prepare_train_prior,
    run_train_carrier_assets as _run_train_carrier_assets,
    run_train_calibrators as _run_train_calibrators,
    run_train_donor as _run_train_donor,
    run_train_lifecycle as _run_train_lifecycle,
    run_train_link_gate as _run_train_link_gate,
    run_train_component_state as _run_train_component_state,
    run_train_spectral_assets as _run_train_spectral_assets,
    run_train_spectra as _run_train_spectra,
    run_train_tokens as _run_train_tokens,
)
from .pipeline.validation import run_validation


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
    "train-tokens",
    "train-component-state",
    "train-donor",
    "train-carrier-assets",
    "train-spectral-assets",
    "train-lifecycle",
    "build-oof-endpoints",
    "build-oof-joint-field",
    "train-link-gate",
    "train-all",
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
    return torch.device(value or f"cuda:{gpu_ids [0 ]}")


def _paths(args: argparse.Namespace) -> RuntimePaths:
    return RuntimePaths.create(
        project_root=PATHS.root,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        output=Path(args.output) if args.output else None,
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
    parser.add_argument("--component-batch-size", type=int, default=32)
    parser.add_argument("--donor-batch-size", type=int, default=128)
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
                {"validate": run_validation(paths)},
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
        reports["infer"] = _run_infer(paths, config, device, args.batch_size)
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
    if args.phase in ("prepare-train-prior", "train-spectral-all", "train-all"):
        reports["prepare_train_prior"] = _run_prepare_train_prior(
            paths, config, gpu_ids, args.force, args.workers_per_gpu
        )
    if args.phase in ("train-spectra", "train-spectral-all", "train-all"):
        reports["train_spectra"] = _run_train_spectra(
            paths,
            config,
            gpu_ids,
            args.force,
            args.workers_per_gpu,
        )
    if args.phase in ("train-calibrators", "train-spectral-all", "train-all"):
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
    if args.phase in ("train-tokens", "train-all"):
        reports["train_tokens"] = _run_train_tokens(paths, config, args.force)
    if args.phase in ("train-component-state", "train-all"):
        reports["train_component_state"] = _run_train_component_state(
            paths,
            config,
            device,
            args.batch_size,
            args.component_batch_size,
            args.force,
        )
    if args.phase in ("train-donor", "train-all"):
        reports["train_donor"] = _run_train_donor(
            paths,
            config,
            device,
            args.batch_size,
            args.donor_batch_size,
            args.force,
        )
    if args.phase in ("train-carrier-assets", "train-all"):
        reports["train_carrier_assets"] = _run_train_carrier_assets(
            paths, config, device, args.batch_size, args.force
        )
    if args.phase in ("train-spectral-assets", "train-all"):
        reports["train_spectral_assets"] = _run_train_spectral_assets(
            paths, config, device, args.batch_size, args.force
        )
    if args.phase in ("train-lifecycle", "train-all"):
        reports["train_lifecycle"] = _run_train_lifecycle(
            paths, config, device, args.batch_size, args.force
        )
    if args.phase in ("build-oof-endpoints", "train-all"):
        reports["build_oof_endpoints"] = _run_build_oof_endpoints(
            paths, config, device, args.batch_size, args.force
        )
    if args.phase in ("build-oof-joint-field", "train-all"):
        reports["build_oof_joint_field"] = _run_build_oof_joint_field(
            paths, device, args.batch_size, args.force
        )
    if args.phase in ("train-link-gate", "train-all"):
        reports["train_link_gate"] = _run_train_link_gate(
            paths, config, device, args.batch_size, args.force
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
