#!/usr/bin/env python3
"""Stage entry point for the self-built Round 2 physical-prior module."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np

from project_paths import PATHS
from round2.data import load_positions
from round2.physical_prior.config import PhysicalPriorConfig
from round2.physical_prior.geometry import (
    build_geometry,
    geometry_summary,
    load_geometry,
    save_geometry,
)
from round2.physical_prior.pipeline import (
    ANALYSIS_DIR,
    GEOMETRY_CACHE,
    RAW_TOKEN_CACHE,
    aggregate_folds,
    atomic_json,
    train_fold,
)
from round2.physical_prior.tokens import (
    assemble_raw_tokens,
    build_raw_token_shard,
)
from round2.physical_prior.tracer import (
    ARTIFACT_DIR,
    WarpTracer,
    build_base_route_bank,
    build_scatter_bank,
    run_trace_shard,
)


PROFILE_REPORT = ANALYSIS_DIR / "performance_profile.json"
ASSEMBLY_REPORT = ANALYSIS_DIR / "raw_token_assembly.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "geometry",
            "profile",
            "trace",
            "route",
            "tokens",
            "assemble",
            "fold",
            "aggregate",
        ),
        required=True,
    )
    parser.add_argument("--site", type=int, default=0)
    parser.add_argument("--branch", choices=("base", "scatter015"), default="base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--outer", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-shard-size", type=int, default=250)
    parser.add_argument("--profile-rays", type=int, default=65_536)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def geometry_stage(config: PhysicalPriorConfig, force: bool) -> dict[str, object]:
    if GEOMETRY_CACHE.is_file() and not force:
        geometry = load_geometry(GEOMETRY_CACHE)
        return {"status": "complete", "reused": True, **geometry_summary(geometry)}
    started = time.time()
    geometry = build_geometry(config)
    save_geometry(GEOMETRY_CACHE, geometry, config)
    return {
        "status": "complete",
        "reused": False,
        "seconds": float(time.time() - started),
        **geometry_summary(geometry),
    }


def profile_stage(
    config: PhysicalPriorConfig, device: str, rays: int
) -> dict[str, object]:
    setup, _, _, _, _ = load_positions()
    geometry_started = time.time()
    geometry = load_geometry(GEOMETRY_CACHE)
    geometry_seconds = time.time() - geometry_started
    bvh_started = time.time()
    tracer = WarpTracer(geometry, device)
    bvh_seconds = time.time() - bvh_started
    branches = []
    for branch in config.branches:
        local = replace(branch, rays_per_site_seed=rays, shard_size=rays)
        arrays, timing = tracer.trace(
            source=setup.bs_positions[0],
            seed=config.seeds[0],
            ray_offset=0,
            ray_count=rays,
            branch=local,
            config=config,
        )
        branches.append(
            {
                "branch": branch.name,
                "rays": rays,
                "hit_fraction": float(np.mean(arrays["count"] > 0)),
                "interaction_count_mean": float(np.mean(arrays["count"])),
                "output_bytes": int(sum(value.nbytes for value in arrays.values())),
                **timing,
                "rays_per_gpu_second": float(
                    rays / max(timing["gpu_trace_seconds"], 1e-9)
                ),
            }
        )
    trace_seconds = sum(
        row["gpu_trace_seconds"] * config.base.rays_per_site_seed / rays
        for row in branches
    )
    expected_parallel = trace_seconds * len(config.seeds) * 2 / 8.0
    report = {
        "status": "complete",
        "device": device,
        "geometry_load_seconds": float(geometry_seconds),
        "bvh_build_seconds": float(bvh_seconds),
        "branches": branches,
        "expected_parallel_trace_seconds": float(expected_parallel),
        "expected_dominant_bottleneck": "GPU repeated mesh intersection and host trace transfer",
        "cpu_preprocessing": "surface-family cache is built once and reused",
        "host_to_device": "mesh and face-family arrays copied once per worker",
        "scoring": "five independent ranker folds run after token construction",
    }
    atomic_json(PROFILE_REPORT, report)
    return report


def main() -> None:
    args = parse_args()
    config = PhysicalPriorConfig()
    if args.stage == "geometry":
        report = geometry_stage(config, args.force)
    elif args.stage == "profile":
        report = profile_stage(config, args.device, args.profile_rays)
    else:
        if not GEOMETRY_CACHE.is_file():
            raise FileNotFoundError(GEOMETRY_CACHE)
        geometry = load_geometry(GEOMETRY_CACHE)
        setup, positions, _, site, _ = load_positions()
        if args.stage == "trace":
            report = run_trace_shard(
                geometry=geometry,
                bs_positions=setup.bs_positions,
                config=config,
                site=args.site,
                branch_name=args.branch,
                seed=args.seed,
                shard=args.shard,
                device=args.device,
                force=args.force,
            )
        elif args.stage == "route":
            if args.branch == config.base.name:
                report = build_base_route_bank(
                    args.site, args.seed, config, force=args.force
                )
            else:
                report = build_scatter_bank(
                    args.site, args.seed, config, force=args.force
                )
        elif args.stage == "tokens":
            point_ids = np.flatnonzero(site == args.site).astype(np.int64)
            report = build_raw_token_shard(
                positions=positions,
                point_ids=point_ids,
                site=args.site,
                bs_positions=setup.bs_positions,
                geometry=geometry,
                config=config,
                shard=args.shard,
                shard_size=args.token_shard_size,
                device=args.device,
                force=args.force,
            )
        elif args.stage == "assemble":
            site_point_ids = {
                int(value): np.flatnonzero(site == value).astype(np.int64)
                for value in np.unique(site)
            }
            report = assemble_raw_tokens(
                site_point_ids,
                positions,
                site,
                setup.bs_positions,
                config,
                args.token_shard_size,
                RAW_TOKEN_CACHE,
            )
            atomic_json(ASSEMBLY_REPORT, report)
        elif args.stage == "fold":
            report = train_fold(
                args.outer,
                geometry,
                config,
                args.device,
                force=args.force,
            )
        elif args.stage == "aggregate":
            report = aggregate_folds(geometry, config)
        else:
            raise ValueError(args.stage)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
