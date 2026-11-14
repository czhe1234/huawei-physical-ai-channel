#!/usr/bin/env python3
"""Render compact canonical-strict E190 path spectra for E196."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any

import numpy as np
import torch

import analyze_v4_path_track_spectral_targets as track_targets
import train_e195_directional_probe_calibrator as e195
import train_surface_interaction_energy_model as e85
from project_paths import PATHS


VERSION = "Round2-E196-canonical-strict-path-spectra"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e196_mechanism_calibrator" / "path_sources"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e196_mechanism_calibrator" / "path_sources"
POINTS = 4000
PATHS_PER_POINT = e85.PATHS_PER_POINT
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=int, choices=range(5), required=True)
    parser.add_argument("--mode", choices=("screen", "refit"), required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def output_paths(outer: int, mode: str) -> dict[str, Path]:
    prefix = ARTIFACT_DIR / f"outer{outer}_{mode}"
    return {
        "pas": prefix.with_name(prefix.name + "_pas_f16.npy"),
        "pdp": prefix.with_name(prefix.name + "_pdp_f16.npy"),
        "ad": prefix.with_name(prefix.name + "_ad_f16.npy"),
        "path_count": prefix.with_name(prefix.name + "_path_count_f32.npy"),
        "report": ANALYSIS_DIR / f"outer{outer}_{mode}.json",
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_memmap(path: Path, dtype: np.dtype, shape: tuple[int, ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    value = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
    return temporary, value


def checkpoint_path(outer: int, mode: str) -> Path:
    name = "screen_base" if mode == "screen" else "refit_base"
    return e195.fold_paths(outer)[name]


def mapping_latent(outer: int, mode: str) -> np.ndarray:
    mapping = np.load(e195.fold_paths(outer)["mapping"], allow_pickle=False)
    return mapping[f"{mode}_latent"].astype(np.float32)


def validate_outputs(paths: dict[str, Path]) -> bool:
    expected = {
        "pas": ((POINTS, 4, 2, 16, 8), np.float16),
        "pdp": ((POINTS, 2, 4, 192), np.float16),
        "ad": ((POINTS, 16, 8, 192), np.float16),
        "path_count": ((POINTS,), np.float32),
    }
    for name, (shape, dtype) in expected.items():
        if not paths[name].is_file():
            return False
        value = np.load(paths[name], mmap_mode="r")
        if value.shape != shape or value.dtype != np.dtype(dtype):
            return False
    return paths["report"].is_file()


@torch.inference_mode()
def render(
    mass: np.ndarray,
    coordinate: np.ndarray,
    port: np.ndarray,
    device: torch.device,
    batch_size: int,
    paths: dict[str, Path],
) -> dict[str, Any]:
    temporary_pas, pas_output = atomic_memmap(
        paths["pas"], np.float16, (POINTS, 4, 2, 16, 8)
    )
    temporary_pdp, pdp_output = atomic_memmap(
        paths["pdp"], np.float16, (POINTS, 2, 4, 192)
    )
    temporary_ad, ad_output = atomic_memmap(
        paths["ad"], np.float16, (POINTS, 16, 8, 192)
    )
    temporary_count, count_output = atomic_memmap(
        paths["path_count"], np.float32, (POINTS,)
    )
    maximum_normalization_error = 0.0
    for start in range(0, POINTS, batch_size):
        end = min(start + batch_size, POINTS)
        local_mass = torch.as_tensor(mass[start:end], device=device)
        local_coordinate = torch.as_tensor(coordinate[start:end], device=device)
        local_port = torch.as_tensor(port[start:end], device=device)
        energy = local_mass[..., None, None] * local_port
        horizontal, vertical, delay = track_targets.spectral_kernels(local_coordinate)
        pas = torch.einsum("bkpu,bkh,bkv->buphv", energy, horizontal, vertical)
        pas = pas / pas.sum(dim=(2, 3, 4), keepdim=True).clamp_min(TINY)
        pdp = torch.einsum("bkpu,bkd->bpud", energy, delay)
        pdp = pdp / pdp.sum(dim=3, keepdim=True).clamp_min(TINY)
        collapsed = energy.sum(dim=(2, 3))
        ad = torch.einsum(
            "bk,bkh,bkv,bkd->bhvd", collapsed, horizontal, vertical, delay
        )
        ad = ad / ad.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)
        active = local_mass > 1e-6
        count_output[start:end] = active.sum(dim=1).float().cpu().numpy()
        pas_output[start:end] = pas.cpu().numpy().astype(np.float16)
        pdp_output[start:end] = pdp.cpu().numpy().astype(np.float16)
        ad_output[start:end] = ad.cpu().numpy().astype(np.float16)
        maximum_normalization_error = max(
            maximum_normalization_error,
            float((pas.sum(dim=(2, 3, 4)) - 1.0).abs().max().item()),
            float((pdp.sum(dim=3) - 1.0).abs().max().item()),
            float((ad.sum(dim=(1, 2, 3)) - 1.0).abs().max().item()),
        )
        if end == POINTS or end % 512 < batch_size:
            for value in (pas_output, pdp_output, ad_output, count_output):
                value.flush()
            print(f"E196 strict path spectra {end}/{POINTS}", flush=True)
    for value in (pas_output, pdp_output, ad_output, count_output):
        value.flush()
    del pas_output, pdp_output, ad_output, count_output
    for temporary, target in (
        (temporary_pas, paths["pas"]),
        (temporary_pdp, paths["pdp"]),
        (temporary_ad, paths["ad"]),
        (temporary_count, paths["path_count"]),
    ):
        os.replace(temporary, target)
    return {"maximum_normalization_error": maximum_normalization_error}


def main() -> None:
    args = parse_args()
    paths = output_paths(args.outer, args.mode)
    if validate_outputs(paths) and not args.force:
        print(paths["report"].read_text(encoding="utf-8"), flush=True)
        return
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    status = "complete"
    error = None
    try:
        checkpoint_file = checkpoint_path(args.outer, args.mode)
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        expected_validation = (args.outer + 1) % 5
        if int(checkpoint["outer"]) != args.outer:
            raise RuntimeError("strict path checkpoint outer mismatch")
        if int(checkpoint["validation_fold"]) != expected_validation:
            raise RuntimeError("strict path checkpoint validation mismatch")
        store = e195.ResidentStore(device)
        coordinate = e195.candidate_coordinates(mapping_latent(args.outer, args.mode), device)
        store.set_mode(coordinate, np.empty((0,), dtype=np.float16))
        model = e85.SurfaceInteractionModel(
            store.base.shape[-1],
            store.sequence_geometry.shape[-1],
            e195.model_config(checkpoint),
            "geometry",
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        mean = torch.as_tensor(
            checkpoint["feature_mean"], device=device, dtype=torch.float32
        )
        std = torch.as_tensor(
            checkpoint["feature_std"], device=device, dtype=torch.float32
        )
        rows = np.arange(POINTS, dtype=np.int64)
        mass, predicted_coordinate, port = e195.endpoint_arrays(
            model,
            store,
            rows,
            mean,
            std,
            args.batch_size,
            include_probe=False,
        )
        render_report = render(
            mass, predicted_coordinate, port, device, args.batch_size, paths
        )
        report = {
            "experiment": VERSION,
            "status": "complete",
            "outer": args.outer,
            "mode": args.mode,
            "checkpoint": PATHS.relative(checkpoint_file),
            "checkpoint_epoch": int(checkpoint["best_epoch"]),
            "mapping": PATHS.relative(e195.fold_paths(args.outer)["mapping"]),
            "outputs": {name: PATHS.relative(path) for name, path in paths.items() if name != "report"},
            "render": render_report,
            "data_lineage": {
                "official_train_h_used_only_in_checkpoint_training": True,
                "historical_complete_channel_read": False,
                "historical_test_endpoint_read": False,
                "outer_label_overlap": 0,
            },
            "performance": {
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "peak_torch_memory_gib": float(torch.cuda.max_memory_allocated(device) / 2**30)
                if device.type == "cuda"
                else 0.0,
                "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
                "wall_seconds": float(time.time() - started),
                "measured_bottleneck": "strict E190 path-set forward and compact AD rendering",
            },
        }
        atomic_json(paths["report"], report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        status = "failed"
        error = traceback.format_exc()
        atomic_json(
            paths["report"],
            {
                "experiment": VERSION,
                "status": status,
                "outer": args.outer,
                "mode": args.mode,
                "traceback": error,
                "wall_seconds": float(time.time() - started),
            },
        )
        raise


if __name__ == "__main__":
    main()
