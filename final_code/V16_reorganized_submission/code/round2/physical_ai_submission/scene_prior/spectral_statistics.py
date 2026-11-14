from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any

import numpy as np
import torch

from ..config import PipelineConfig, RuntimePaths
from ..data.official_data import load_official_data
from ..data.artifact_io import atomic_json, atomic_npz


TINY = 1e-30
DOMAINS = ("pas", "pdp", "ad", "log_power", "component_count")
CFR_SCHEMA_VERSION = "carrier_phase_v1"


@dataclass(frozen=True)
class SpectralTask:
    split: str
    configuration: str
    seed: int
    site: int
    cuda_device: int
    support_dir: str
    output_dir: str
    carrier_frequency_hz: float
    force: bool


def _statistics(channel: torch.Tensor) -> dict[str, np.ndarray]:
    count = len(channel)
    shaped = channel.reshape(count, 2, 16, 8, 4, 192)
    spatial = torch.fft.fft2(shaped, dim=(2, 3))
    pas_power = spatial.abs().square().float()
    pas_total = pas_power.sum(dim=(1, 2, 3))
    pas_valid = pas_total > TINY
    pas = torch.where(
        pas_valid[:, None, None, None],
        pas_power / pas_total[:, None, None, None].clamp_min(TINY),
        0.0,
    )
    delay = torch.fft.ifft(channel, dim=-1)
    pdp_power = delay.abs().square().float()
    pdp_total = pdp_power.sum(dim=-1)
    pdp_valid = pdp_total > TINY
    pdp = torch.where(
        pdp_valid[..., None],
        pdp_power / pdp_total[..., None].clamp_min(TINY),
        0.0,
    )
    joint = torch.fft.ifft(spatial, dim=-1)
    ad_power = joint.abs().square().sum(dim=(1, 4), dtype=torch.float64).float()
    ad_total = ad_power.sum(dim=(1, 2, 3))
    ad_valid = ad_total > TINY
    ad = torch.where(
        ad_valid[:, None, None, None],
        ad_power / ad_total[:, None, None, None].clamp_min(TINY),
        0.0,
    )
    point_power = channel.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    point_valid = point_power > TINY
    log_power = torch.where(
        point_valid, torch.log(point_power.clamp_min(TINY)), 0.0
    ).float()
    return {
        "pas": pas.cpu().numpy().astype(np.float32),
        "pas_valid": pas_valid.cpu().numpy(),
        "pdp": pdp.cpu().numpy().astype(np.float32),
        "pdp_valid": pdp_valid.cpu().numpy(),
        "ad": ad.cpu().numpy().astype(np.float32),
        "ad_valid": ad_valid.cpu().numpy(),
        "log_power": log_power.cpu().numpy().astype(np.float32),
        "log_power_valid": point_valid.cpu().numpy(),
    }


@torch.inference_mode()
def _channel_from_support(
    loaded: np.lib.npyio.NpzFile,
    device: torch.device,
    carrier_frequency_hz: float,
) -> torch.Tensor:
    offsets = np.asarray(loaded["offsets"], dtype=np.int64)
    frequencies = torch.arange(-96, 96, dtype=torch.float32, device=device) * 30_000.0
    channels: list[torch.Tensor] = []
    for row in range(len(offsets) - 1):
        start, end = int(offsets[row]), int(offsets[row + 1])
        if end <= start:
            channels.append(
                torch.zeros((256, 4, 192), dtype=torch.complex64, device=device)
            )
            continue
        coefficient = torch.as_tensor(
            np.asarray(loaded["full_complex_coefficient"][start:end]), device=device
        )
        tau = torch.as_tensor(
            np.asarray(loaded["tau"][start:end]), dtype=torch.float32, device=device
        )

        carrier_phase = (
            torch.remainder(
                -2.0 * torch.pi * tau * carrier_frequency_hz + torch.pi,
                2.0 * torch.pi,
            )
            - torch.pi
        )
        carrier_phasor = torch.complex(
            torch.cos(carrier_phase), torch.sin(carrier_phase)
        )
        baseband_phase = (
            torch.remainder(
                -2.0 * torch.pi * tau[:, None] * frequencies[None] + torch.pi,
                2.0 * torch.pi,
            )
            - torch.pi
        )
        baseband_phasor = torch.complex(
            torch.cos(baseband_phase), torch.sin(baseband_phase)
        )
        baseband_coefficient = coefficient * carrier_phasor[:, None, None]
        channels.append(
            torch.einsum("pmu,ps->mus", baseband_coefficient, baseband_phasor)
        )
    return torch.stack(channels)


def _spectral_shard_path(output_dir: Path, support_path: Path) -> Path:
    return output_dir / support_path.name


def _valid_spectral_shard(path: Path, carrier_frequency_hz: float) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return bool(
                str(loaded["cfr_schema_version"].item()) == CFR_SCHEMA_VERSION
                and float(loaded["carrier_frequency_hz"].item())
                == float(carrier_frequency_hz)
            )
    except (OSError, ValueError, KeyError):
        return False


def _run_task(task: SpectralTask) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(task.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = output_dir / "failure.json"
    failure_path.unlink(missing_ok=True)
    try:
        device = torch.device(f"cuda:{task .cuda_device }")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        support_paths = sorted(Path(task.support_dir).glob("batch_*.npz"))
        if not support_paths:
            raise FileNotFoundError(f"no support shards in {task .support_dir }")
        completed = reused = components_total = 0
        for support_path in support_paths:
            output = _spectral_shard_path(output_dir, support_path)
            if not task.force and _valid_spectral_shard(
                output, task.carrier_frequency_hz
            ):
                reused += 1
                continue
            with np.load(support_path, allow_pickle=False) as loaded:
                channel = _channel_from_support(
                    loaded, device, task.carrier_frequency_hz
                )
                statistics = _statistics(channel)
                statistics["point_ids"] = np.asarray(
                    loaded["point_ids"], dtype=np.int64
                )
                statistics["component_count"] = np.diff(
                    np.asarray(loaded["offsets"], dtype=np.int64)
                ).astype(np.float32)
                statistics["cfr_schema_version"] = np.asarray(CFR_SCHEMA_VERSION)
                statistics["carrier_frequency_hz"] = np.asarray(
                    task.carrier_frequency_hz, dtype=np.float64
                )
                components_total += int(np.sum(statistics["component_count"]))
            atomic_npz(output, **statistics)
            completed += 1
            del channel
        report = {
            "status": "complete",
            "configuration": task.configuration,
            "seed": task.seed,
            "site": task.site,
            "cuda_device": task.cuda_device,
            "completed_batches": completed,
            "reused_batches": reused,
            "component_count": components_total,
            "cfr_schema_version": CFR_SCHEMA_VERSION,
            "carrier_frequency_hz": task.carrier_frequency_hz,
            "peak_torch_memory_gib": float(
                torch.cuda.max_memory_allocated(device) / 2**30
            ),
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "wall_seconds": float(time.time() - started),
        }
        atomic_json(output_dir / "task_report.json", report)
        return report
    except BaseException as error:
        report = {
            "status": "failed",
            "configuration": task.configuration,
            "seed": task.seed,
            "site": task.site,
            "cuda_device": task.cuda_device,
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
            "wall_seconds": float(time.time() - started),
        }
        atomic_json(failure_path, report)
        return report


def _run_queue(tasks: tuple[SpectralTask, ...]) -> list[dict[str, Any]]:
    return [_run_task(task) for task in tasks]


def _tasks(
    paths: RuntimePaths,
    config: PipelineConfig,
    split: str,
    gpu_ids: tuple[int, ...],
    force: bool,
) -> list[SpectralTask]:
    tasks: list[SpectralTask] = []
    index = 0
    for configuration in ("base", "scatter"):
        for seed in config.seeds:
            for site in (0, 1):
                gpu = gpu_ids[index % len(gpu_ids)]
                support = (
                    paths.prior_support_dir
                    / split
                    / configuration
                    / f"seed_{seed :04d}"
                    / f"site_{site }"
                )
                output = (
                    paths.work_dir
                    / "physical_spectra"
                    / "shards"
                    / split
                    / configuration
                    / f"seed_{seed :04d}"
                    / f"site_{site }"
                )
                tasks.append(
                    SpectralTask(
                        split=split,
                        configuration=configuration,
                        seed=seed,
                        site=site,
                        cuda_device=gpu,
                        support_dir=str(support),
                        output_dir=str(output),
                        carrier_frequency_hz=config.physical_carrier_frequency_hz,
                        force=force,
                    )
                )
                index += 1
    return tasks


def build_spectral_shards(
    paths: RuntimePaths,
    config: PipelineConfig,
    split: str,
    gpu_ids: tuple[int, ...],
    force: bool = False,
    max_tasks: int | None = None,
    workers_per_gpu: int = 2,
) -> dict[str, Any]:
    started = time.time()
    tasks = _tasks(paths, config, split, gpu_ids, force)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    queues: list[tuple[SpectralTask, ...]] = []
    for gpu in gpu_ids:
        local = tuple(task for task in tasks if task.cuda_device == gpu)
        lanes = min(workers_per_gpu, len(local))
        queues.extend(tuple(local[lane::lanes]) for lane in range(lanes))
    queues = [queue for queue in queues if queue]
    context = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(queues), mp_context=context) as pool:
        futures = [pool.submit(_run_queue, queue) for queue in queues]
        for future in as_completed(futures):
            results.extend(future.result())
    failures = [result for result in results if result["status"] != "complete"]
    report = {
        "stage": "physical_spectral_shards",
        "status": "failed" if failures else "complete",
        "split": split,
        "task_count": len(tasks),
        "failed_task_count": len(failures),
        "gpu_ids": list(gpu_ids),
        "workers_per_gpu": workers_per_gpu,
        "wall_seconds": float(time.time() - started),
        "measured_bottleneck": "GPU component-to-CFR synthesis and FFT",
        "failures": failures,
    }
    atomic_json(paths.report_dir / f"physical_spectral_shards_{split }.json", report)
    if failures:
        raise RuntimeError(f"{len (failures )} physical spectral tasks failed")
    return report


def _domain_shapes(count: int) -> dict[str, tuple[int, ...]]:
    return {
        "pas": (count, 2, 16, 8, 4, 192),
        "pdp": (count, 256, 4, 192),
        "ad": (count, 16, 8, 192),
        "log_power": (count,),
        "component_count": (count,),
    }


def _count_shapes(count: int) -> dict[str, tuple[int, ...]]:
    return {
        "pas": (count, 4, 192),
        "pdp": (count, 256, 4),
        "ad": (count,),
        "log_power": (count,),
        "component_count": (count,),
    }


def _expand_valid(domain: str, value: np.ndarray) -> np.ndarray:
    if domain == "pas":
        return value[:, None, None, None]
    if domain == "pdp":
        return value[..., None]
    if domain == "ad":
        return value[:, None, None, None]
    return value


def _aggregate_configuration(
    paths: RuntimePaths,
    config: PipelineConfig,
    split: str,
    count: int,
    configuration: str,
    force: bool = False,
) -> dict[str, Any]:
    started = time.time()
    shapes = _domain_shapes(count)
    count_shapes = _count_shapes(count)
    root = paths.work_dir / "physical_spectra" / split / configuration
    report_path = root / "aggregate_report.json"
    outputs = {
        domain: {
            kind: root / f"{domain }_{kind }.npy" for kind in ("mean", "m2", "count")
        }
        for domain in DOMAINS
    }
    cached_report: dict[str, Any] = {}
    if report_path.is_file():
        try:
            cached_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached_report = {}
    if (
        not force
        and cached_report.get("cfr_schema_version") == CFR_SCHEMA_VERSION
        and float(cached_report.get("carrier_frequency_hz", -1.0))
        == float(config.physical_carrier_frequency_hz)
        and all(
            path.is_file() for values in outputs.values() for path in values.values()
        )
    ):
        return cached_report | {"status": "reused"}

    root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, dict[str, np.memmap]] = {}
    partials: list[tuple[Path, Path]] = []
    for domain in DOMAINS:
        arrays[domain] = {}
        for kind, dtype, shape in (
            ("mean", np.float32, shapes[domain]),
            ("m2", np.float32, shapes[domain]),
            ("count", np.uint8, count_shapes[domain]),
        ):
            final = outputs[domain][kind]
            partial = final.with_suffix(final.suffix + ".partial.npy")
            partial.unlink(missing_ok=True)
            value = np.lib.format.open_memmap(
                partial, mode="w+", dtype=dtype, shape=shape
            )
            value[...] = 0
            arrays[domain][kind] = value
            partials.append((partial, final))
    for seed in config.seeds:
        for site in (0, 1):
            shard_root = (
                paths.work_dir
                / "physical_spectra"
                / "shards"
                / split
                / configuration
                / f"seed_{seed :04d}"
                / f"site_{site }"
            )
            shards = sorted(shard_root.glob("batch_*.npz"))
            if not shards:
                raise FileNotFoundError(f"missing spectral shards in {shard_root }")
            for shard in shards:
                with np.load(shard, allow_pickle=False) as loaded:
                    rows = np.asarray(loaded["point_ids"], dtype=np.int64)
                    for domain in DOMAINS:
                        value = np.asarray(loaded[domain], dtype=np.float32)
                        valid = (
                            np.ones(len(rows), dtype=bool)
                            if domain == "component_count"
                            else np.asarray(loaded[f"{domain }_valid"], dtype=bool)
                        )
                        mean = arrays[domain]["mean"][rows]
                        m2 = arrays[domain]["m2"][rows]
                        old_count = arrays[domain]["count"][rows].astype(np.float32)
                        new_count = old_count + valid.astype(np.float32)
                        expanded_valid = _expand_valid(domain, valid)
                        expanded_count = _expand_valid(
                            domain, np.maximum(new_count, 1.0)
                        )
                        delta = value - mean
                        updated = np.where(
                            expanded_valid,
                            mean + delta / expanded_count,
                            mean,
                        )
                        m2 = np.where(
                            expanded_valid,
                            m2 + delta * (value - updated),
                            m2,
                        )
                        arrays[domain]["mean"][rows] = updated.astype(np.float32)
                        arrays[domain]["m2"][rows] = m2.astype(np.float32)
                        arrays[domain]["count"][rows] = new_count.astype(np.uint8)
    for values in arrays.values():
        for value in values.values():
            value.flush()
    del arrays
    for partial, final in partials:
        os.replace(partial, final)
    local_report = {
        "status": "complete",
        "configuration": configuration,
        "split": split,
        "seed_count": config.seed_count,
        "cfr_schema_version": CFR_SCHEMA_VERSION,
        "carrier_frequency_hz": config.physical_carrier_frequency_hz,
        "wall_seconds": float(time.time() - started),
    }
    atomic_json(report_path, local_report)
    return local_report


def aggregate_spectral_statistics(
    paths: RuntimePaths,
    config: PipelineConfig,
    split: str,
    force: bool = False,
) -> dict[str, Any]:
    started = time.time()
    data = load_official_data(paths.data_dir)
    count = len(data.train_positions) if split == "train" else len(data.test_positions)
    configuration_reports: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        futures = {
            pool.submit(
                _aggregate_configuration,
                paths,
                config,
                split,
                count,
                configuration,
                force,
            ): configuration
            for configuration in ("base", "scatter")
        }
        for future in as_completed(futures):
            configuration = futures[future]
            try:
                configuration_reports[configuration] = future.result()
            except BaseException as error:
                failures.append(
                    {
                        "configuration": configuration,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
    report = {
        "stage": "physical_spectral_aggregation",
        "status": "failed" if failures else "complete",
        "split": split,
        "configurations": configuration_reports,
        "failures": failures,
        "wall_seconds": float(time.time() - started),
        "measured_bottleneck": "parallel CPU decompression, Welford, and disk writes",
    }
    atomic_json(
        paths.report_dir / f"physical_spectral_aggregation_{split }.json", report
    )
    if failures:
        raise RuntimeError(f"{len (failures )} spectral aggregations failed")
    return report
